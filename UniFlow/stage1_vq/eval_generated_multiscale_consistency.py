import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import models.vqvae as vqvae
import utils.utils_model as utils_model
from stage1_vq.eval_single_scale_latent import build_stage1_args


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-encode real/generated arrays with a multiscale Stage-1 tokenizer "
            "and measure whether different scale branches recover consistent public ids."
        )
    )
    parser.add_argument("--vqvae_ckpt", type=str, required=True)
    parser.add_argument("--array_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--array_names",
        nargs="+",
        default=["real", "shared_subspace", "scale_0_subspace", "scale_1_subspace"],
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def load_model(vqvae_ckpt, device):
    checkpoint = torch.load(vqvae_ckpt, map_location="cpu", weights_only=False)
    if "net" not in checkpoint:
        raise KeyError(f"Expected stage1 checkpoint with key 'net', got keys: {sorted(checkpoint.keys())}")
    state_dict = checkpoint["net"]
    stage1_args, run_log_path = build_stage1_args(
        argparse.Namespace(
            vqvae_ckpt=vqvae_ckpt,
            dataname=None,
            window_size=None,
            down_t=None,
            quantizer=None,
            num_workers=0,
        ),
        state_dict,
    )
    num_codes, code_dim = state_dict["vqvae.quantizer.codebook"].shape

    model = vqvae.VQVAE(
        stage1_args,
        num_codes,
        code_dim,
        stage1_args.down_t,
        stage1_args.stride_t,
        stage1_args.width,
        stage1_args.depth,
        stage1_args.dilation_growth_rate,
        stage1_args.vq_act,
        stage1_args.vq_norm,
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, stage1_args, run_log_path


def top_margin(similarity):
    if similarity.shape[-1] < 2:
        return torch.zeros(similarity.shape[:-1], device=similarity.device)
    values = torch.topk(similarity, k=2, dim=-1).values
    return values[..., 0] - values[..., 1]


@torch.no_grad()
def encode_consistency_batch(model, batch):
    shared_idx = model.encode(batch)
    scale_latents = model.encode_scale_latents(batch)
    inner = model.vqvae

    scale_indices = []
    branch_margins = []
    shared_similarity = None

    weights = getattr(inner.quantizer, "scale_weights_tensor", None)
    if weights is None:
        weights = torch.ones(len(scale_latents), device=batch.device) / len(scale_latents)
    weights = weights.to(batch.device)

    for scale_id, scale_latent in enumerate(scale_latents):
        batch_size, seq_len, latent_dim = scale_latent.shape
        scale_quantizer = inner.quantizer.scale_quantizers[scale_id]
        flat_latent = torch.nn.functional.normalize(
            scale_latent.contiguous().view(-1, latent_dim),
            dim=-1,
        )
        similarity = scale_quantizer.compute_similarity(flat_latent)
        scale_idx = similarity.argmax(dim=-1).view(batch_size, seq_len)
        scale_indices.append(scale_idx)
        branch_margins.append(top_margin(similarity).view(batch_size, seq_len))
        weighted_similarity = weights[scale_id] * similarity
        shared_similarity = (
            weighted_similarity
            if shared_similarity is None
            else shared_similarity + weighted_similarity
        )

    shared_margin = top_margin(shared_similarity).view_as(shared_idx)
    return shared_idx, scale_indices, shared_margin, branch_margins


def init_stats(num_scales):
    return {
        "samples": 0,
        "tokens": 0,
        "pair_token_match": 0.0,
        "pair_sequence_match": 0.0,
        "both_match_shared_token": 0.0,
        "both_match_shared_sequence": 0.0,
        "scale_to_shared_token": [0.0 for _ in range(num_scales)],
        "scale_to_shared_sequence": [0.0 for _ in range(num_scales)],
        "shared_margin_sum": 0.0,
        "branch_margin_sum": [0.0 for _ in range(num_scales)],
        "used_codes_shared": set(),
        "used_codes_scale": [set() for _ in range(num_scales)],
    }


def update_stats(stats, shared_idx, scale_indices, shared_margin, branch_margins):
    batch_size, seq_len = shared_idx.shape
    stats["samples"] += batch_size
    stats["tokens"] += batch_size * seq_len
    stats["used_codes_shared"].update(shared_idx.detach().cpu().numpy().reshape(-1).tolist())

    if len(scale_indices) >= 2:
        pair_match = scale_indices[0] == scale_indices[1]
        stats["pair_token_match"] += pair_match.sum().item()
        stats["pair_sequence_match"] += pair_match.all(dim=1).sum().item()

    both_match = torch.ones_like(shared_idx, dtype=torch.bool)
    for scale_id, scale_idx in enumerate(scale_indices):
        match_shared = scale_idx == shared_idx
        stats["scale_to_shared_token"][scale_id] += match_shared.sum().item()
        stats["scale_to_shared_sequence"][scale_id] += match_shared.all(dim=1).sum().item()
        both_match = both_match & match_shared
        stats["used_codes_scale"][scale_id].update(
            scale_idx.detach().cpu().numpy().reshape(-1).tolist()
        )

    stats["both_match_shared_token"] += both_match.sum().item()
    stats["both_match_shared_sequence"] += both_match.all(dim=1).sum().item()
    stats["shared_margin_sum"] += shared_margin.sum().item()
    for scale_id, margin in enumerate(branch_margins):
        stats["branch_margin_sum"][scale_id] += margin.sum().item()


def finalize_stats(stats):
    tokens = max(stats["tokens"], 1)
    samples = max(stats["samples"], 1)
    result = {
        "samples": int(stats["samples"]),
        "tokens": int(stats["tokens"]),
        "pair_token_match_acc": float(stats["pair_token_match"] / tokens),
        "pair_sequence_exact_acc": float(stats["pair_sequence_match"] / samples),
        "both_match_shared_token_acc": float(stats["both_match_shared_token"] / tokens),
        "both_match_shared_sequence_acc": float(stats["both_match_shared_sequence"] / samples),
        "shared_margin_mean": float(stats["shared_margin_sum"] / tokens),
        "used_codes_shared": int(len(stats["used_codes_shared"])),
    }
    for scale_id, token_sum in enumerate(stats["scale_to_shared_token"]):
        result[f"scale_{scale_id}_to_shared_token_acc"] = float(token_sum / tokens)
        result[f"scale_{scale_id}_to_shared_sequence_acc"] = float(
            stats["scale_to_shared_sequence"][scale_id] / samples
        )
        result[f"scale_{scale_id}_margin_mean"] = float(
            stats["branch_margin_sum"][scale_id] / tokens
        )
        result[f"used_codes_scale_{scale_id}"] = int(len(stats["used_codes_scale"][scale_id]))
    return result


def evaluate_array(model, array_path, batch_size, device, max_samples):
    arr = np.load(array_path, mmap_mode="r")
    total = arr.shape[0] if max_samples is None else min(arr.shape[0], max_samples)
    if total <= 0:
        raise ValueError(f"No samples in {array_path}")

    num_scales = len(getattr(model.vqvae, "scale_dims", [model.vqvae.code_dim]))
    stats = init_stats(num_scales)

    for start in tqdm(range(0, total, batch_size), desc=os.path.basename(array_path), ncols=100):
        end = min(start + batch_size, total)
        batch_np = np.asarray(arr[start:end], dtype=np.float32).copy()
        batch = torch.from_numpy(batch_np).to(device)
        shared_idx, scale_indices, shared_margin, branch_margins = encode_consistency_batch(
            model, batch
        )
        update_stats(stats, shared_idx, scale_indices, shared_margin, branch_margins)

    return finalize_stats(stats), list(arr.shape[1:])


def write_markdown(path, results):
    order = [name for name in ["real", "shared_subspace", "scale_0_subspace", "scale_1_subspace"] if name in results["metrics"]]
    order += [name for name in results["metrics"] if name not in order]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Generated Multiscale Consistency\n\n")
        handle.write(
            "This evaluates generated arrays by re-encoding each sequence with the Stage-1 "
            "multiscale tokenizer and checking whether different scale branches recover "
            "consistent public code ids.\n\n"
        )
        handle.write("| Array | Pair Token Acc | Pair Seq Acc | Both-to-Shared Token Acc | Shared Margin | Used Shared Codes |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for name in order:
            item = results["metrics"][name]
            handle.write(
                f"| {name} | {item['pair_token_match_acc']:.4f} | "
                f"{item['pair_sequence_exact_acc']:.4f} | "
                f"{item['both_match_shared_token_acc']:.4f} | "
                f"{item['shared_margin_mean']:.6f} | "
                f"{item['used_codes_shared']} |\n"
            )
        handle.write("\n")
        handle.write("| Array | Scale-1 to Shared | Scale-2 to Shared | Scale-1 Margin | Scale-2 Margin |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        for name in order:
            item = results["metrics"][name]
            handle.write(
                f"| {name} | {item.get('scale_0_to_shared_token_acc', 0.0):.4f} | "
                f"{item.get('scale_1_to_shared_token_acc', 0.0):.4f} | "
                f"{item.get('scale_0_margin_mean', 0.0):.6f} | "
                f"{item.get('scale_1_margin_mean', 0.0):.6f} |\n"
            )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vqvae_ckpt = os.path.abspath(args.vqvae_ckpt)
    array_dir = os.path.abspath(args.array_dir)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(array_dir, "multiscale_consistency")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger = utils_model.get_logger(output_dir)

    model, stage1_args, run_log_path = load_model(vqvae_ckpt, device)
    scale_dims = list(getattr(model.vqvae, "scale_dims", [model.vqvae.code_dim]))
    logger.info(f"Loaded VQVAE: {vqvae_ckpt}")
    logger.info(f"Stage1 run log: {run_log_path}")
    logger.info(f"Scale dims: {scale_dims}")

    results = {
        "config": {
            "vqvae_ckpt": vqvae_ckpt,
            "array_dir": array_dir,
            "output_dir": output_dir,
            "device": str(device),
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "stage1_config": vars(stage1_args),
            "scale_dims": scale_dims,
        },
        "metrics": {},
    }

    for array_name in args.array_names:
        array_path = os.path.join(array_dir, f"{array_name}.npy")
        if not os.path.exists(array_path):
            logger.info(f"Skipping missing array: {array_path}")
            continue
        metrics, sample_shape = evaluate_array(
            model,
            array_path,
            args.batch_size,
            device,
            args.max_samples,
        )
        metrics["sample_shape"] = sample_shape
        results["metrics"][array_name] = metrics
        logger.info(f"{array_name}: {json.dumps(metrics, sort_keys=True)}")

    if not results["metrics"]:
        raise FileNotFoundError(f"No requested arrays found in {array_dir}")

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    markdown_path = os.path.join(output_dir, "summary.md")
    write_markdown(markdown_path, results)
    logger.info(f"Saved summary to {summary_path}")
    logger.info(f"Saved markdown to {markdown_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset import dataset_VQ
from models.uniflow import KDEPrior, UniFlowModel
import models.vqvae as vqvae
import utils.utils_model as utils_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate stage2 generation by keeping only one multiscale latent subspace "
            "during low-rank sampling and code-id retrieval."
        )
    )
    parser.add_argument("--uniflow_ckpt", "--sdflow_ckpt", dest="uniflow_ckpt",
                        type=str, required=True, help="path to Stage-2 best_ds.pth")
    parser.add_argument("--vqvae_ckpt", type=str, default=None, help="override stage1 ckpt from stage2 args")
    parser.add_argument("--output_dir", type=str, default=None, help="default: <stage2_ckpt_dir>/single_scale_subspace_eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_steps", type=int, default=None, help="override stage2 eval_steps")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--dataset_type", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--selected_scales", nargs="+", type=int, default=None, help="default: evaluate every scale branch")
    parser.add_argument("--metrics", type=str, default="basic", choices=["basic", "all"])
    parser.add_argument("--store_arrays", action="store_true", help="store full real/generated arrays even when metrics='basic'")
    parser.add_argument("--max_samples", type=int, default=None, help="truncate evaluation set for quick checks")
    parser.add_argument("--save_preview_count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampler", type=str, default="kde", choices=["kde", "mixup"])
    parser.add_argument("--kde_bandwidth_factor", type=float, default=0.001)
    return parser.parse_args()


def extract_first_json_block(text):
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def load_stage1_config_from_log(vqvae_ckpt):
    run_log_path = os.path.join(os.path.dirname(os.path.abspath(vqvae_ckpt)), "run.log")
    if not os.path.exists(run_log_path):
        return None, run_log_path

    with open(run_log_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    json_block = extract_first_json_block(text)
    if json_block is None:
        return None, run_log_path

    return json.loads(json_block), run_log_path


def resolve_checkpoint_path(path_value):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(REPO_ROOT, path_value))


def build_stage1_args(vqvae_ckpt, checkpoint_state):
    config_dict, run_log_path = load_stage1_config_from_log(vqvae_ckpt)
    if config_dict is None:
        raise FileNotFoundError(
            "Could not recover stage1 config from run.log next to the VQ checkpoint. "
            f"Expected a parsable log at: {run_log_path}"
        )

    defaults = {
        "stride_t": 2,
        "width": 512,
        "depth": 3,
        "dilation_growth_rate": 3,
        "vq_act": "relu",
        "vq_norm": None,
        "mu": 0.99,
        "multi_scale_vq": False,
        "multi_scale_mode": "receptive",
        "multi_scale_dims": None,
        "multi_scale_depths": None,
        "multi_scale_dilations": None,
        "multi_scale_weights": None,
        "multi_scale_patches": None,
        "multi_scale_patch_downs": None,
    }
    for key, value in defaults.items():
        config_dict.setdefault(key, value)

    required_keys = ["dataname", "window_size", "down_t", "quantizer"]
    missing = [key for key in required_keys if key not in config_dict or config_dict[key] is None]
    if missing:
        raise ValueError(
            f"Stage1 config is missing required keys {missing}. "
            "Please restore run.log or pass a matching stage1 checkpoint."
        )

    nb_code, code_dim = checkpoint_state["vqvae.quantizer.codebook"].shape
    config_dict["nb_code"] = int(nb_code)
    config_dict["code_dim"] = int(code_dim)

    return argparse.Namespace(**config_dict), run_log_path


def make_variant_specs(scale_dims, selected_scales):
    num_scales = len(scale_dims)
    if selected_scales is None:
        scale_ids = list(range(num_scales))
    else:
        scale_ids = sorted({int(scale_id) for scale_id in selected_scales})

    for scale_id in scale_ids:
        if scale_id < 0 or scale_id >= num_scales:
            raise ValueError(f"Scale id {scale_id} is out of range for {num_scales} branches")

    variant_specs = [("shared_subspace", None)]
    for scale_id in scale_ids:
        variant_specs.append((f"scale_{scale_id}_subspace", [scale_id]))
    return variant_specs


def build_scale_mask(scale_dims, selected_scales, device, dtype):
    if selected_scales is None:
        return None

    mask = torch.zeros(sum(scale_dims), device=device, dtype=dtype)
    offset = 0
    selected = set(int(scale_id) for scale_id in selected_scales)
    for scale_id, dim in enumerate(scale_dims):
        if scale_id in selected:
            mask[offset : offset + dim] = 1.0
        offset += dim
    return mask.view(1, 1, -1)


def sample_u(model, batch_size, sampler, device, kde_solver=None):
    if sampler == "kde":
        if kde_solver is None:
            raise ValueError("kde solver is required when sampler='kde'")
        return kde_solver.sample(batch_size)

    idx_a = torch.randint(0, model.num_samples, (batch_size,), device=device)
    idx_b = torch.randint(0, model.num_samples, (batch_size,), device=device)
    u_a = model.U[idx_a]
    u_b = model.U[idx_b]
    alpha = torch.rand(batch_size, 1, device=device)
    u_mix = alpha * u_a + (1.0 - alpha) * u_b
    noise = torch.randn_like(u_mix) * 0.01
    return u_mix + noise


@torch.no_grad()
def generate_indices_for_variant(
    model,
    codebook,
    u_sample,
    seq_len,
    steps,
    temperature,
    scale_mask=None,
):
    batch_size = u_sample.shape[0]
    z0_flat = torch.matmul(u_sample, model.V)
    z = z0_flat.reshape(batch_size, model.seq_len, model.code_dim)
    z = F.normalize(z, dim=-1)

    masked_codebook = codebook
    if scale_mask is not None:
        z = z * scale_mask
        masked_codebook = codebook * scale_mask.view(1, -1)

    t_span = torch.linspace(0, 1, steps + 1, device=z.device)
    if model.t_scheduler == "cosine":
        t_span = 1 - torch.cos((t_span ** 2.0) * 0.5 * torch.pi)

    for index in range(len(t_span) - 1):
        t_curr = t_span[index]
        dt = t_span[index + 1] - t_curr
        t_batch = t_curr.unsqueeze(0).repeat(batch_size)

        logits = model.forward(z, t_batch)
        probs = F.softmax(logits / temperature, dim=-1)
        mu_t = torch.matmul(probs, masked_codebook)

        if scale_mask is not None:
            mu_t = mu_t * scale_mask

        if t_curr < 0.999:
            velocity = (mu_t - z) / (1 - t_curr + 1e-5)
        else:
            velocity = torch.zeros_like(z)
        z = z + velocity * dt

        if scale_mask is not None:
            z = z * scale_mask

    z = F.normalize(z, dim=-1)
    if scale_mask is not None:
        z = z * scale_mask

    z_flat = z.reshape(-1, model.code_dim)
    distances = torch.cdist(z_flat, masked_codebook)
    indices = distances.argmin(dim=-1).reshape(batch_size, seq_len)
    return indices


def init_array_storage(output_dir, variant_names, total_samples, sample_shape):
    array_paths = {}
    arrays = {}
    for name in ["real"] + list(variant_names):
        path = os.path.join(output_dir, f"{name}.npy")
        arrays[name] = open_memmap(path, mode="w+", dtype=np.float32, shape=(total_samples,) + sample_shape)
        array_paths[name] = path
    return arrays, array_paths


def update_error_stats(stats, real_np, preds_np):
    diff = preds_np.astype(np.float64, copy=False) - real_np.astype(np.float64, copy=False)
    stats["sq_sum"] += np.square(diff).sum()
    stats["abs_sum"] += np.abs(diff).sum()
    stats["count"] += diff.size


def finalize_basic_metrics(stats):
    denominator = max(stats["count"], 1)
    return {
        "mse": float(stats["sq_sum"] / denominator),
        "mae": float(stats["abs_sum"] / denominator),
    }


def count_preview_samples(parts):
    return sum(part.shape[0] for part in parts)


def compute_advanced_metrics(real_np, preds_np):
    from metrics.context_fid import Context_FID
    from metrics.discriminative_metrics import discriminative_score_metrics
    from metrics.predictive_metrics import predictive_score_metrics2 as predictive_score_metrics

    metrics = {}

    ds_scores = []
    for _ in range(3):
        ds_scores.append(float(discriminative_score_metrics(real_np, preds_np)))
    metrics["ds_mean"] = float(np.mean(ds_scores))
    metrics["ds_std"] = float(np.std(ds_scores))

    ps_scores = []
    for _ in range(3):
        ps_scores.append(float(predictive_score_metrics(real_np, preds_np)))
    metrics["ps_mean"] = float(np.mean(ps_scores))
    metrics["ps_std"] = float(np.std(ps_scores))

    real_fid = np.transpose(real_np, (0, 2, 1))
    pred_fid = np.transpose(preds_np, (0, 2, 1))
    fid_scores = []
    for _ in range(3):
        fid_scores.append(float(Context_FID(real_fid, pred_fid)))
    metrics["fid_mean"] = float(np.mean(fid_scores))
    metrics["fid_std"] = float(np.std(fid_scores))

    return metrics


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    uniflow_ckpt = os.path.abspath(args.uniflow_ckpt)
    if not os.path.exists(uniflow_ckpt):
        raise FileNotFoundError(f"Stage2 checkpoint not found: {uniflow_ckpt}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(uniflow_ckpt), "single_scale_subspace_eval")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger = utils_model.get_logger(output_dir)

    uniflow_checkpoint = torch.load(uniflow_ckpt, map_location="cpu", weights_only=False)
    if "model_state_dict" not in uniflow_checkpoint:
        raise KeyError(
            f"Expected stage2 checkpoint with key 'model_state_dict', got keys: {sorted(uniflow_checkpoint.keys())}"
        )
    stage2_args = uniflow_checkpoint["args"]

    vqvae_ckpt = resolve_checkpoint_path(args.vqvae_ckpt or stage2_args["vqvae_ckpt"])
    if vqvae_ckpt is None or not os.path.exists(vqvae_ckpt):
        raise FileNotFoundError(f"Stage1 checkpoint not found: {vqvae_ckpt}")

    vqvae_checkpoint = torch.load(vqvae_ckpt, map_location="cpu", weights_only=False)
    if "net" not in vqvae_checkpoint:
        raise KeyError(f"Expected stage1 checkpoint with key 'net', got keys: {sorted(vqvae_checkpoint.keys())}")
    vq_state_dict = vqvae_checkpoint["net"]
    num_codes, code_dim = vq_state_dict["vqvae.quantizer.codebook"].shape

    stage1_args, run_log_path = build_stage1_args(vqvae_ckpt, vq_state_dict)

    logger.info(
        json.dumps(
            {
                "uniflow_ckpt": uniflow_ckpt,
                "vqvae_ckpt": vqvae_ckpt,
                "output_dir": output_dir,
                "device": str(device),
                "metrics": args.metrics,
                "store_arrays": args.store_arrays,
                "max_samples": args.max_samples,
                "selected_scales": args.selected_scales,
                "sampler": args.sampler,
                "kde_bandwidth_factor": args.kde_bandwidth_factor,
                "stage2_args": stage2_args,
                "stage1_run_log_path": run_log_path,
                "resolved_stage1_config": vars(stage1_args),
            },
            indent=2,
            sort_keys=True,
        )
    )

    vq_model = vqvae.VQVAE(
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
    vq_model.load_state_dict(vq_state_dict, strict=True)
    vq_model.eval()

    real_loader = dataset_VQ.DATALoader(
        stage2_args["dataname"],
        batch_size=args.batch_size,
        num_workers=0,
        window_size=stage2_args["window_size"],
        unit_length=2 ** stage2_args["down_t"],
        dataset_type=args.dataset_type,
    )

    dataset_size = len(real_loader.dataset)
    total_samples = dataset_size if args.max_samples is None else min(dataset_size, args.max_samples)
    if total_samples <= 0:
        raise ValueError("No samples available for evaluation")

    codebook = vq_state_dict["vqvae.quantizer.codebook"].to(device)
    seq_len = stage2_args["window_size"] // (2 ** stage2_args["down_t"])

    model = UniFlowModel(
        num_samples=uniflow_checkpoint["model_state_dict"]["U"].shape[0],
        num_codes=num_codes,
        code_dim=code_dim,
        seq_len=seq_len,
        rank=stage2_args["rank"],
        d_model=stage2_args["d_model"],
        n_layers=stage2_args["n_layers"],
        num_heads=stage2_args["num_heads"],
        dropout=stage2_args["dropout"],
        noise_std=stage2_args["noise_std"],
        lambda_mean=stage2_args["lambda_mean"],
        lambda_std=stage2_args["lambda_std"],
        t_scheduler="cosine",
    ).to(device)
    model.load_state_dict(uniflow_checkpoint["model_state_dict"], strict=True)
    model.eval()

    kde_solver = None
    if args.sampler == "kde":
        kde_solver = KDEPrior(model.U, device=device, bandwidth_factor=args.kde_bandwidth_factor)

    inner_vq = vq_model.vqvae
    scale_dims = list(getattr(inner_vq, "scale_dims", [inner_vq.code_dim]))
    variant_specs = make_variant_specs(scale_dims, args.selected_scales)
    variant_names = [name for name, _ in variant_specs]
    scale_masks = {
        name: build_scale_mask(scale_dims, selected, device=device, dtype=codebook.dtype)
        for name, selected in variant_specs
    }

    logger.info(f"Evaluating variants: {variant_specs}")
    logger.info(f"Scale dims: {scale_dims}")

    should_store_full_arrays = args.metrics == "all" or args.store_arrays
    stored_arrays = None
    stored_array_paths = None
    basic_stats = {
        name: {"sq_sum": 0.0, "abs_sum": 0.0, "count": 0}
        for name in variant_names
    }
    preview = {"real": []}
    for name in variant_names:
        preview[name] = []
    used_codes = {name: set() for name in variant_names}
    eval_steps = stage2_args["eval_steps"] if args.eval_steps is None else args.eval_steps

    num_written = 0
    with torch.no_grad():
        progress = tqdm(total=total_samples, desc="Generating", ncols=100)
        for real_batch in real_loader:
            if num_written >= total_samples:
                break

            real_batch = real_batch.float()
            if real_batch.shape[0] > total_samples - num_written:
                real_batch = real_batch[: total_samples - num_written]
            real_batch = real_batch.to(device)
            real_np = real_batch.detach().cpu().numpy().astype(np.float32, copy=False)

            if stored_arrays is None and should_store_full_arrays:
                sample_shape = tuple(real_np.shape[1:])
                stored_arrays, stored_array_paths = init_array_storage(
                    output_dir, variant_names, total_samples, sample_shape
                )

            batch_size = real_np.shape[0]
            u_sample = sample_u(
                model,
                batch_size=batch_size,
                sampler=args.sampler,
                device=device,
                kde_solver=kde_solver,
            )

            batch_outputs = {}
            for name, selected in variant_specs:
                indices = generate_indices_for_variant(
                    model=model,
                    codebook=codebook,
                    u_sample=u_sample,
                    seq_len=seq_len,
                    steps=eval_steps,
                    temperature=args.temperature,
                    scale_mask=scale_masks[name],
                )
                used_codes[name].update(indices.detach().cpu().numpy().reshape(-1).tolist())
                recon = vq_model.forward_decoder(indices)
                batch_outputs[name] = recon.detach().cpu().numpy().astype(np.float32, copy=False)

            slice_end = num_written + batch_size
            if stored_arrays is not None:
                stored_arrays["real"][num_written:slice_end] = real_np

            preview_slots = max(
                args.save_preview_count - count_preview_samples(preview["real"]),
                0,
            )
            if preview_slots > 0:
                preview["real"].append(real_np[:preview_slots].copy())

            for name, preds_np in batch_outputs.items():
                update_error_stats(basic_stats[name], real_np, preds_np)
                if stored_arrays is not None:
                    stored_arrays[name][num_written:slice_end] = preds_np
                if preview_slots > 0:
                    preview[name].append(preds_np[:preview_slots].copy())

            num_written = slice_end
            progress.update(batch_size)
        progress.close()

    results = {
        "config": {
            "uniflow_ckpt": uniflow_ckpt,
            "vqvae_ckpt": vqvae_ckpt,
            "output_dir": output_dir,
            "dataset_type": args.dataset_type,
            "evaluated_samples": num_written,
            "metrics": args.metrics,
            "sampler": args.sampler,
            "eval_steps": eval_steps,
            "temperature": args.temperature,
            "scale_dims": scale_dims,
            "variant_specs": [
                {"name": name, "selected_scales": selected}
                for name, selected in variant_specs
            ],
            "used_code_count": {
                name: len(code_ids) for name, code_ids in used_codes.items()
            },
        },
        "metrics": {},
    }

    for name in variant_names:
        results["metrics"][name] = finalize_basic_metrics(basic_stats[name])

    preview_to_save = {}
    for name, parts in preview.items():
        if not parts:
            continue
        preview_to_save[name] = np.concatenate(parts, axis=0)
    if preview_to_save:
        preview_path = os.path.join(output_dir, "preview.npz")
        np.savez_compressed(preview_path, **preview_to_save)
        results["config"]["preview_path"] = preview_path

    if stored_arrays is not None:
        for array in stored_arrays.values():
            array.flush()
        results["config"]["array_paths"] = stored_array_paths

    if args.metrics == "all":
        real_np = np.load(stored_array_paths["real"], mmap_mode="r")
        for name in variant_names:
            logger.info(f"Computing advanced metrics for {name} ...")
            preds_np = np.load(stored_array_paths[name], mmap_mode="r")
            results["metrics"][name].update(compute_advanced_metrics(real_np, preds_np))

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    for name, metrics_dict in results["metrics"].items():
        logger.info(f"{name}: {json.dumps(metrics_dict, sort_keys=True)}")
    logger.info(f"Used codes: {results['config']['used_code_count']}")
    logger.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

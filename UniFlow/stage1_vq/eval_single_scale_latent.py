import argparse
import json
import os
import sys

import numpy as np
import torch
from numpy.lib.format import open_memmap
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset import dataset_VQ
import models.vqvae as vqvae
import utils.utils_model as utils_model


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether a single-scale latent branch can recover the shared code id, "
            "then decode the full multiscale latent using that id."
        )
    )
    parser.add_argument("--vqvae_ckpt", type=str, required=True, help="path to stage1 net_best_ds.pth")
    parser.add_argument("--output_dir", type=str, default=None, help="default: <ckpt_dir>/single_scale_eval")
    parser.add_argument("--dataname", type=str, default=None, help="override dataname from run.log")
    parser.add_argument("--dataset_type", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--selected_scales", nargs="+", type=int, default=None, help="default: evaluate every scale branch")
    parser.add_argument("--metrics", type=str, default="basic", choices=["basic", "all"])
    parser.add_argument("--store_arrays", action="store_true", help="store full labels/prediction arrays even when metrics='basic'")
    parser.add_argument("--max_samples", type=int, default=None, help="truncate evaluation set for quick checks")
    parser.add_argument("--save_preview_count", type=int, default=16, help="save the first N samples into preview.npz")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--window_size", type=int, default=None, help="override window_size from run.log")
    parser.add_argument("--down_t", type=int, default=None, help="override down_t from run.log")
    parser.add_argument("--quantizer", type=str, default=None, help="override quantizer from run.log")
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


def build_stage1_args(eval_args, checkpoint_state):
    config_dict, run_log_path = load_stage1_config_from_log(eval_args.vqvae_ckpt)
    if config_dict is None:
        raise FileNotFoundError(
            "Could not recover stage1 config from run.log next to the checkpoint. "
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

    if eval_args.dataname is not None:
        config_dict["dataname"] = eval_args.dataname
    if eval_args.window_size is not None:
        config_dict["window_size"] = eval_args.window_size
    if eval_args.down_t is not None:
        config_dict["down_t"] = eval_args.down_t
    if eval_args.quantizer is not None:
        config_dict["quantizer"] = eval_args.quantizer
    if eval_args.num_workers is not None:
        config_dict["num_workers"] = eval_args.num_workers

    required_keys = ["dataname", "window_size", "down_t", "quantizer"]
    missing = [key for key in required_keys if key not in config_dict or config_dict[key] is None]
    if missing:
        raise ValueError(
            f"Stage1 config is missing required keys {missing}. "
            "Please provide them explicitly on the CLI."
        )

    nb_code, code_dim = checkpoint_state["vqvae.quantizer.codebook"].shape
    config_dict["nb_code"] = int(nb_code)
    config_dict["code_dim"] = int(code_dim)

    return argparse.Namespace(**config_dict), run_log_path


def make_variant_specs(model, selected_scales):
    inner_model = model.vqvae
    if getattr(inner_model, "use_multiscale", False):
        num_scales = len(inner_model.scale_dims)
    else:
        num_scales = 1

    if selected_scales is None:
        scale_ids = list(range(num_scales))
    else:
        scale_ids = sorted({int(scale_id) for scale_id in selected_scales})

    for scale_id in scale_ids:
        if scale_id < 0 or scale_id >= num_scales:
            raise ValueError(f"Scale id {scale_id} is out of range for {num_scales} branches")

    variant_specs = [("shared_id", None)]
    for scale_id in scale_ids:
        variant_specs.append((f"scale_{scale_id}_id", [scale_id]))
    return variant_specs


def init_array_storage(output_dir, variant_names, total_samples, sample_shape):
    array_paths = {}
    arrays = {}
    for name in ["labels"] + list(variant_names):
        path = os.path.join(output_dir, f"{name}.npy")
        arrays[name] = open_memmap(path, mode="w+", dtype=np.float32, shape=(total_samples,) + sample_shape)
        array_paths[name] = path
    return arrays, array_paths


def update_error_stats(stats, labels_np, preds_np):
    diff = preds_np.astype(np.float64, copy=False) - labels_np.astype(np.float64, copy=False)
    stats["sq_sum"] += np.square(diff).sum()
    stats["abs_sum"] += np.abs(diff).sum()
    stats["count"] += diff.size


def update_match_stats(stats, shared_indices, variant_indices):
    matches = (shared_indices == variant_indices)
    stats["token_match"] += matches.sum().item()
    stats["token_total"] += matches.numel()
    stats["sequence_match"] += matches.all(dim=1).sum().item()
    stats["sequence_total"] += matches.shape[0]


def finalize_basic_metrics(stats):
    denominator = max(stats["count"], 1)
    token_total = max(stats["token_total"], 1)
    sequence_total = max(stats["sequence_total"], 1)
    return {
        "mse": float(stats["sq_sum"] / denominator),
        "mae": float(stats["abs_sum"] / denominator),
        "token_match_acc": float(stats["token_match"] / token_total),
        "sequence_exact_acc": float(stats["sequence_match"] / sequence_total),
    }


def compute_advanced_metrics(labels_np, preds_np):
    from metrics.context_fid import Context_FID
    from metrics.discriminative_metrics import discriminative_score_metrics
    from metrics.predictive_metrics import predictive_score_metrics2 as predictive_score_metrics

    metrics = {}

    ds_scores = []
    for _ in range(3):
        ds_scores.append(float(discriminative_score_metrics(labels_np, preds_np)))
    metrics["ds_mean"] = float(np.mean(ds_scores))
    metrics["ds_std"] = float(np.std(ds_scores))

    ps_scores = []
    for _ in range(3):
        ps_scores.append(float(predictive_score_metrics(labels_np, preds_np)))
    metrics["ps_mean"] = float(np.mean(ps_scores))
    metrics["ps_std"] = float(np.std(ps_scores))

    real_fid = np.transpose(labels_np, (0, 2, 1))
    pred_fid = np.transpose(preds_np, (0, 2, 1))
    fid_scores = []
    for _ in range(3):
        fid_scores.append(float(Context_FID(real_fid, pred_fid)))
    metrics["fid_mean"] = float(np.mean(fid_scores))
    metrics["fid_std"] = float(np.std(fid_scores))

    return metrics


def count_preview_samples(parts):
    return sum(part.shape[0] for part in parts)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.abspath(args.vqvae_ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(ckpt_path), "single_scale_eval")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger = utils_model.get_logger(output_dir)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "net" not in checkpoint:
        raise KeyError(f"Expected stage1 checkpoint with key 'net', got keys: {sorted(checkpoint.keys())}")
    state_dict = checkpoint["net"]

    stage1_args, run_log_path = build_stage1_args(args, state_dict)
    num_codes, code_dim = state_dict["vqvae.quantizer.codebook"].shape

    logger.info(
        json.dumps(
            {
                "vqvae_ckpt": ckpt_path,
                "output_dir": output_dir,
                "device": str(device),
                "metrics": args.metrics,
                "store_arrays": args.store_arrays,
                "max_samples": args.max_samples,
                "selected_scales": args.selected_scales,
                "resolved_stage1_config": vars(stage1_args),
                "run_log_path": run_log_path,
            },
            indent=2,
            sort_keys=True,
        )
    )

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

    data_loader = dataset_VQ.DATALoader(
        stage1_args.dataname,
        batch_size=args.batch_size,
        num_workers=stage1_args.num_workers,
        window_size=stage1_args.window_size,
        unit_length=2 ** stage1_args.down_t,
        dataset_type=args.dataset_type,
    )

    dataset_size = len(data_loader.dataset)
    total_samples = dataset_size if args.max_samples is None else min(dataset_size, args.max_samples)
    if total_samples <= 0:
        raise ValueError("No samples available for evaluation")

    variant_specs = make_variant_specs(model, args.selected_scales)
    variant_names = [name for name, _ in variant_specs]

    inner_model = model.vqvae
    scale_dims = list(getattr(inner_model, "scale_dims", [inner_model.code_dim]))
    logger.info(f"Evaluating variants: {variant_specs}")
    logger.info(f"Scale dims: {scale_dims}")

    should_store_full_arrays = args.metrics == "all" or args.store_arrays
    stored_arrays = None
    stored_array_paths = None
    basic_stats = {
        name: {
            "sq_sum": 0.0,
            "abs_sum": 0.0,
            "count": 0,
            "token_match": 0.0,
            "token_total": 0,
            "sequence_match": 0.0,
            "sequence_total": 0,
        }
        for name in variant_names
    }
    preview = {"labels": []}
    for name in variant_names:
        preview[name] = []

    used_codes = {name: set() for name in variant_names}
    num_written = 0

    with torch.no_grad():
        progress = tqdm(total=total_samples, desc="Evaluating", ncols=100)
        for batch in data_loader:
            if num_written >= total_samples:
                break

            batch = batch.float()
            if batch.shape[0] > total_samples - num_written:
                batch = batch[: total_samples - num_written]
            batch = batch.to(device)

            labels_np = batch.detach().cpu().numpy().astype(np.float32, copy=False)

            if stored_arrays is None and should_store_full_arrays:
                sample_shape = tuple(labels_np.shape[1:])
                stored_arrays, stored_array_paths = init_array_storage(
                    output_dir, variant_names, total_samples, sample_shape
                )

            shared_indices = model.encode(batch)
            scale_latents = model.encode_scale_latents(batch)

            batch_indices = {"shared_id": shared_indices}
            for name, selected in variant_specs[1:]:
                scale_id = selected[0]
                batch_indices[name] = model.quantize_single_scale_latent(
                    scale_latents[scale_id], scale_id=scale_id
                )

            batch_outputs = {}
            for name, indices in batch_indices.items():
                used_codes[name].update(indices.detach().cpu().numpy().reshape(-1).tolist())
                update_match_stats(basic_stats[name], shared_indices, indices)
                batch_outputs[name] = model.forward_decoder(indices)

            batch_size = labels_np.shape[0]
            slice_end = num_written + batch_size

            if stored_arrays is not None:
                stored_arrays["labels"][num_written:slice_end] = labels_np

            preview_slots = max(
                args.save_preview_count - count_preview_samples(preview["labels"]),
                0,
            )
            if preview_slots > 0:
                preview["labels"].append(labels_np[:preview_slots].copy())

            for name, output in batch_outputs.items():
                preds_np = output.detach().cpu().numpy().astype(np.float32, copy=False)
                update_error_stats(basic_stats[name], labels_np, preds_np)

                if stored_arrays is not None:
                    stored_arrays[name][num_written:slice_end] = preds_np

                if preview_slots > 0:
                    preview[name].append(preds_np[:preview_slots].copy())

            num_written = slice_end
            progress.update(batch_size)
        progress.close()

    results = {
        "config": {
            "vqvae_ckpt": ckpt_path,
            "output_dir": output_dir,
            "dataset_type": args.dataset_type,
            "evaluated_samples": num_written,
            "metrics": args.metrics,
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
        labels_np = np.load(stored_array_paths["labels"], mmap_mode="r")
        for name in variant_names:
            logger.info(f"Computing advanced metrics for {name} ...")
            preds_np = np.load(stored_array_paths[name], mmap_mode="r")
            results["metrics"][name].update(compute_advanced_metrics(labels_np, preds_np))

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    logger.info(f"Used codes: {results['config']['used_code_count']}")
    for name, metrics_dict in results["metrics"].items():
        logger.info(f"{name}: {json.dumps(metrics_dict, sort_keys=True)}")
    logger.info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

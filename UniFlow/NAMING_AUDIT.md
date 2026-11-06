# UniFlow Naming & Release Audit

This audit refers to the uploaded ZIP and the prepared GitHub-facing repository. Changes aim to improve public naming **without invalidating existing checkpoints**.

## Canonical names introduced / updated

| Location | Old name | New public-facing name / action |
|---|---|---|
| `README.md` | Minimal `# uniflow` with two commands | Comprehensive `UniFlow` README |
| `environment.yaml` | `name: sdflow` | `name: uniflow` |
| `models/sdflow.py` | SDFlow module and `SDFlowModel` | Core implementation moved to `models/uniflow.py`; `UniFlowModel` canonical; old import is an alias |
| `stage2_flow/train_sdflow.py` | SDFlow training entry | `stage2_flow/train_uniflow.py` canonical; old path is a wrapper |
| `utils/eval_sdflow.py` | Legacy evaluation helper | `utils/eval_uniflow.py` canonical; old import path remains |
| `stage1_vq/train_vq.py` | `utils.eval_sdflow`, `sdflow_tensorboard` | New eval import and `uniflow_tensorboard` |
| `stage2_flow/train_uniflow.py` | SDFlow CLI title, `checkpoints_SDFlow`, TensorBoard prefix | UniFlow labels and defaults |
| `options/option_vq.py`, `options/option_transformer.py` | Descriptions mention SDflow | Descriptions mention UniFlow |
| `scripts/stage2_Flow/*.bat` | SDFlow printed names and entrypoints | UniFlow printed names and entrypoints |
| `stage2_flow/eval_single_scale_generation.py` | `--sdflow_ckpt`, `SDFlowModel` | `--uniflow_ckpt` alias, canonical UniFlow model, legacy CLI flag retained |

## Intentionally retained

- `VQVAE`, `TSVQVAE`, `ema_reset_sim`, `vqvae_ckpt`, and `vqvae.quantizer.*`: established technical terms and stable model/checkpoint keys.
- `KDEPrior`, `GMM`, `TS2Vec`, dataset names, and metrics: algorithm/baseline terminology rather than project branding.
- Existing `.bat` filenames ending in `_gmm`: preserved so external references to historical experiment scripts continue working; their *printed titles* and training entrypoint now use UniFlow.
- Legacy `models.sdflow`, `stage2_flow/train_sdflow.py`, `utils.eval_sdflow`, and `--sdflow_ckpt`: compatibility shims. Avoid removing these until legacy scripts/checkpoints are retired.

## Other GitHub release checks

1. Dataset files are absent. The loader expects fixed paths under `dataset/`.
2. The repository has no `LICENSE` file. Choose a license before release.
3. The Conda environment is legacy (Python 3.8 and a CUDA 11.3 PyTorch wheel); a clean install has not been confirmed.
4. `dataset/dataset_VQ.py` currently takes all CSV rows for training (empty val/test partitions); evaluation split design deserves review before claiming held-out scores.
5. README installation uses `YOUR_USERNAME` as an explicit placeholder; replace it with the actual account name after publishing.
6. No UniFlow paper URL, repository URL, author list, or benchmark statistics have been invented in README.
7. A generated Mermaid architecture diagram is embedded in README; no figure asset is required.

@echo off
REM Evaluate stage1 reconstruction when only one latent scale is kept.

setlocal
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%\..\.."

set NUM_THREADS=1
set OPENBLAS_NUM_THREADS=%NUM_THREADS%

echo ================================
echo Single-Scale Latent Eval
echo ================================
echo.

python stage1_vq\eval_single_scale_latent.py ^
    --vqvae_ckpt ".\output\output_sine_multipatch_weighted\VQVAE_MS_P24_D192_320_W4060\net_best_ds.pth" ^
    --metrics basic ^
    --device cuda

python stage1_vq\eval_single_scale_latent.py ^
    --vqvae_ckpt ".\output\output_stock_multipatch_weighted\VQVAE_MS_P24_D192_320_W4060\net_best_ds.pth" ^
    --metrics basic ^
    --device cuda

python stage1_vq\eval_single_scale_latent.py ^
    --vqvae_ckpt ".\output\output_etth_multipatch_weighted\VQVAE_MS_P24_D192_320_W4060\net_best_ds.pth" ^
    --metrics basic ^
    --device cuda

python stage1_vq\eval_single_scale_latent.py ^
    --vqvae_ckpt ".\output\output_energy_multipatch_weighted\VQVAE_MS_P24_D192_320_W4060\net_best_ds.pth" ^
    --metrics basic ^
    --device cuda

echo.
echo ================================
echo Finished
echo ================================
echo.

popd
pause
endlocal

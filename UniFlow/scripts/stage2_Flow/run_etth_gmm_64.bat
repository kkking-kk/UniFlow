@echo off
REM UniFlow

setlocal
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%\..\.."

set NUM_THREADS=1
set OPENBLAS_NUM_THREADS=%NUM_THREADS%

echo ================================
echo UniFlow
echo ================================
echo.

set VQVAE_CKPT=.\output\output_etth_multipatch_weighted\VQVAE_MS_P64_D192_320_W4060\net_best_ds.pth

echo start training UniFlow model...
echo.

python stage2_flow\train_uniflow.py ^
    --vqvae_ckpt "%VQVAE_CKPT%" ^
    --dataname etth ^
    --output_dir .\checkpoints_etth_ms_64 ^
    --window_size 64 ^
    --down_t 2 ^
    --quantizer ema_reset_sim ^
    --rank 256 ^
    --d_model 1024 ^
    --n_layers 3 ^
    --num_heads 16 ^
    --dropout 0.1 ^
    --noise_std 0.01 ^
    --lambda_mean 0.1 ^
    --lambda_std 10.0 ^
    --batch_size 64 ^
    --lr 1e-4 ^
    --lr_uv 1e-3 ^
    --max_iters 100000 ^
    --print_interval 200 ^
    --eval_interval 20000 ^
    --eval_steps 20 ^
    --device cuda

echo.
echo ================================
echo Finished
echo ================================
echo.

popd
pause
endlocal

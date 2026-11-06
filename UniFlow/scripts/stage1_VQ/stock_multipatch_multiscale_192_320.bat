@echo off
setlocal

set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%\..\.."

python stage1_vq\train_vq.py ^
  --batch-size 64 ^
  --num-workers 0 ^
  --width 512 ^
  --lr 1e-4 ^
  --total-iter 100000 ^
  --lr-scheduler 200000 ^
  --code-dim 512 ^
  --nb-code 256 ^
  --down-t 2 ^
  --depth 3 ^
  --dilation-growth-rate 3 ^
  --out-dir .\output\output_stock_multipatch_weighted ^
  --dataname stock ^
  --vq-act relu ^
  --quantizer ema_reset_sim ^
  --exp-name VQVAE_MS_P24_D192_320_W4060 ^
  --window-size 24 ^
  --commit 2 ^
  --gpu 0 ^
  --multi-scale-vq ^
  --multi-scale-mode patch ^
  --multi-scale-dims 192 320 ^
  --multi-scale-depths 2 4 ^
  --multi-scale-dilations 1 3 ^
  --multi-scale-weights 0.4 0.6 ^
  --multi-scale-patches 2 4 ^
  --multi-scale-patch-downs 1 0

popd
endlocal

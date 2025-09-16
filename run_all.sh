#!/bin/bash

IMAGES=("JasperRidge" "PaviaU")
CASES=(1 2 3 4 5)

for IMG in "${IMAGES[@]}"; do
  for CASE in "${CASES[@]}"; do
    echo "Running: Image=$IMG, Case=$CASE"
    python hsi_test_gst.py \
      -a qrnn3d \
      --dataroot "data/forGeoSSTV/normalized/${IMG}/Case${CASE}" \
      -r \
      -rp checkpoints/qrnn3d/paviaft/model_epoch_150_160454.pth \
      --gpu-ids 0 \
      --prefix "denoise_${IMG}_Case${CASE}"
  done
done

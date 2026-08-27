#!/bin/bash
set -euo pipefail

echo "Installing Python requirements..."
pip3 install -r requirements.txt

echo "Cloning VGGT (feed-forward submap generator; paper Sec. 3)..."
if [ ! -d vggt ]; then
  git clone https://github.com/facebookresearch/vggt.git
fi
pip install -e ./vggt

echo "Installing VGGT-SLAM++..."
pip install -e .

echo "Done. Off-the-shelf DINOv2 weights are fetched on first run via torch.hub."
echo "FAISS-HNSW runs on CPU. Optional SALAD is not required (w_loops = 0)."

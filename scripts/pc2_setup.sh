#!/usr/bin/env bash
# Reproducible PC2 setup (Jetson Orin NX 16 GB, JetPack 6.2 / L4T 36.4.3, Ubuntu 22.04, Python 3.10).
# Verified end to end on 2026-09-26. Idempotent: safe to re-run. Needs sudo for the apt part.
#
#   bash scripts/pc2_setup.sh            # everything
#   bash scripts/pc2_setup.sh --no-apt   # skip the system packages (already installed)
#
# What it does and why (details in docs/01_research.md §5 and docs/04_deploy_checklist.md):
#   1. apt: CUDA 12.6 runtime libraries, cuDNN 9, TensorRT 10.3 runtime + Python bindings, DLA compiler lib.
#      Deliberately NOT the `nvidia-jetpack` metapackage (it pulls nvcc, samples, VPI, Nsight, OpenCV-CUDA: ~4 GB more).
#   2. conda env `g1fetch` as an OFFLINE CLONE of the existing `uni` env (pinocchio 3.1 + CasADi from conda-forge,
#      unitree_sdk2py, cyclonedds 0.10.2). Rebuilding those from scratch is slow and fragile on aarch64.
#   3. pip: pinned packages from requirements-pc2.txt (PyPI), then torch/torchvision from NVIDIA's Jetson index with
#      --no-deps. PyPI's aarch64 torch is a CPU or CUDA-13 SBSA build and does not see the Tegra GPU.
#   4. Environment hooks: CUDA + cuDSS library paths in the env's activate.d, system TensorRT module via a .pth file.
#   5. Verification and TensorRT engine export for the two detectors.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
DO_APT=1
[ "${1:-}" = "--no-apt" ] && DO_APT=0

APT_PKGS=(
  cuda-libraries-12-6=12.6.11-1
  cuda-nvtx-12-6=12.6.68-1
  cuda-cupti-12-6=12.6.68-1
  libcudnn9-cuda-12=9.3.0.75-1
  libnvinfer10=10.3.0.30-1+cuda12.5
  libnvinfer-plugin10=10.3.0.30-1+cuda12.5
  libnvonnxparsers10=10.3.0.30-1+cuda12.5
  libnvinfer-dispatch10=10.3.0.30-1+cuda12.5
  libnvinfer-lean10=10.3.0.30-1+cuda12.5
  python3-libnvinfer=10.3.0.30-1+cuda12.5
  nvidia-l4t-dla-compiler                 # any r36.4 build; provides libnvdla_compiler.so needed by `import tensorrt`
)
JETSON_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126
TORCH_VER=2.11.0
TORCHVISION_VER=0.26.0

if [ "$DO_APT" = 1 ]; then
  echo "== 1. system packages (CUDA runtime, cuDNN, TensorRT) =="
  sudo apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${APT_PKGS[@]}" \
    || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${APT_PKGS[@]%%=*}"
  sudo ldconfig
  /usr/bin/python3 -c "import tensorrt; print('system tensorrt', tensorrt.__version__)"
fi

echo "== 2. conda env g1fetch (offline clone of uni) =="
source ~/miniconda3/etc/profile.d/conda.sh
if ! conda env list | grep -qE "^g1fetch "; then
  conda create -n g1fetch --clone uni --offline -y
fi
conda activate g1fetch

echo "== 3. pip packages =="
pip install -r "$ROOT/requirements-pc2.txt"
# Jetson CUDA builds only from the Jetson index, no PyPI fallback, no dependency resolution (deps already present).
pip install --no-deps --index-url "$JETSON_INDEX" "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER"
# libcudss.so.0 for the Jetson torch build; --no-deps keeps its CUDA 12.9 pip libraries away from the system CUDA 12.6.
pip install --no-deps "nvidia-cudss-cu12==0.8.0.10"
pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 cuda-toolkit 2>/dev/null || true

echo "== 4. environment hooks =="
SP=$(python -c "import site; print(site.getsitepackages()[0])")
CUDSS=$(dirname "$(find "$SP/nvidia" -name 'libcudss.so.0*' | head -1)")
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda.sh" <<EOF
export PATH=/usr/local/cuda-12.6/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$CUDSS:\${LD_LIBRARY_PATH:-}
EOF
echo "/usr/lib/python3.10/dist-packages" > "$SP/system-tensorrt.pth"   # system tensorrt module (same Python 3.10)
conda deactivate && conda activate g1fetch

echo "== 5. verify =="
python - <<'EOF'
import pinocchio, casadi, unitree_sdk2py, pyrealsense2, ultralytics, torch, tensorrt
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("tensorrt", tensorrt.__version__, "| ultralytics", ultralytics.__version__, "| pyrealsense2", pyrealsense2.__version__)
assert torch.cuda.is_available(), "torch does not see the Orin GPU: check the Jetson index install and activate.d/cuda.sh"
EOF

echo "== 6. TensorRT engines (skipped if present; ~8 min each) =="
cd "$ROOT"
if [ ! -f models/yoloe-26s-seg-fridge.pt ] || [ ! -f models/yolo26n.pt ]; then
  echo "copy models/yoloe-26s-seg-fridge.pt and models/yolo26n.pt from the workstation (scripts/export_detector.py --bake) first"
else
  [ -f models/yoloe-26s-seg-fridge.engine ] && [ -f models/yolo26n.engine ] || python scripts/export_detector.py --engine
fi

echo "== done. next: python -m g1_fetch.cli check --config configs/pc2.yaml =="
echo "note: stop the teleop camera stream while using the camera:  sudo systemctl stop teleimager-realsense-webrtc  (start it again afterwards)"

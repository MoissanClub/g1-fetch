#!/usr/bin/env bash
# PC2 (Jetson Orin NX, JetPack 6.2, Ubuntu 22.04) setup for g1-fridge-fetch.
# Run as the normal user; sudo is requested where needed. Idempotent-ish. Read docs/04_deploy_checklist.md first.
set -euo pipefail

PY=${PY:-python3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

echo "== apt packages =="
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv libopencv-dev libboost-all-dev cmake git curl

echo "== python packages (JetPack 6 wheels for torch/torchvision from Ultralytics assets) =="
$PY -m pip install --upgrade pip
$PY -m pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.10.0-cp310-cp310-linux_aarch64.whl \
                   https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.25.0-cp310-cp310-linux_aarch64.whl \
  || echo "torch wheel install failed: check https://docs.ultralytics.com/guides/nvidia-jetson/ for the current JetPack 6 wheels"
$PY -m pip install ultralytics opencv-python-headless pyyaml scipy numpy requests pytest pytest-timeout
$PY -m pip install pin casadi || echo "pinocchio/casadi pip install failed: use conda-forge (pinocchio>=3) instead"

echo "== unitree_sdk2_python =="
if ! $PY -c "import unitree_sdk2py" 2>/dev/null; then
  if [ ! -d "$HOME/unitree_sdk2_python" ]; then
    git clone https://github.com/unitreerobotics/unitree_sdk2_python "$HOME/unitree_sdk2_python"
  fi
  $PY -m pip install -e "$HOME/unitree_sdk2_python"
fi

echo "== librealsense / pyrealsense2 =="
if ! $PY -c "import pyrealsense2" 2>/dev/null; then
  cat <<'EOF'
pyrealsense2 is not installed. On JetPack 6.2 use one of:
  a) JetsonHacks prebuilt kernel modules + librealsense:  https://github.com/jetsonhacks/jetson-orin-librealsense
  b) Source build:  git clone https://github.com/IntelRealSense/librealsense && cd librealsense && mkdir build && cd build \
     && cmake .. -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$(which python3) -DCMAKE_BUILD_TYPE=Release \
     && make -j6 && sudo make install   # then add /usr/local/lib/python3.10/... to PYTHONPATH if needed
EOF
fi

echo "== stop the teleop camera pipeline while autonomy runs (single V4L2 owner) =="
sudo systemctl stop teleimager-webrtc 2>/dev/null || true
sudo systemctl stop teleimager 2>/dev/null || true

echo "== detector engines =="
echo "copy models/yoloe-26s-seg-fridge.pt and models/yolo26n.pt into $ROOT/models, then:"
echo "  $PY $ROOT/scripts/export_detector.py --engine"

echo "== checks =="
echo "  cd $ROOT && $PY -m g1_fetch.cli check"

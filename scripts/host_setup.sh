#!/usr/bin/env bash
# Workstation (AMD64, CPU only) environment: conda env `g1fetch` cloned from `uni` + pinned pip packages.
# Import `pinocchio` before `torch` in this env (conda libstdc++ must load first); tests/conftest.py does that.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda env list | grep -qE "^g1fetch " || conda create -n g1fetch --clone uni -y
conda activate g1fetch
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0" "torchvision==0.29.0"
pip install -r "$ROOT/requirements-host.txt"
pip install -e "$ROOT/../unitree_sdk2_python"
python - <<'EOF'
import pinocchio, casadi, torch, ultralytics, cv2, unitree_sdk2py
print("ok: pinocchio", pinocchio.__version__, "torch", torch.__version__, "ultralytics", ultralytics.__version__)
EOF
echo "run: cd $ROOT && python -m pytest tests -q"

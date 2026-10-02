# 06 · Dependency installation on PC2: issues, decisions, learnings

For agents setting up GPU inference on this robot's Jetson Orin NX (JetPack 6.2 / L4T 36.4.3, Python 3.10) or a
similar Jetson. Each item: what happened → what I decided → the rule to follow. The reproducible result is
`scripts/pc2_setup.sh` + `requirements-pc2.txt`; read those first, this file explains why they look the way they do.

## 1. "JetPack is installed" told me nothing about CUDA
- **Happened:** the user said JetPack 6.2 was installed. True for the L4T base, but `apt-cache policy nvidia-jetpack`
  showed `Installed: (none)`, there was no `/usr/local/cuda`, `import tensorrt` failed, and the existing conda env's
  torch reported `cuda=False`. Detector inference ran at 2.5–5 s per frame on the CPU.
- **Decision:** verify, never infer. Four commands settle it: `apt-cache policy nvidia-jetpack`, `ls /usr/local/cuda*`,
  `python3 -c "import tensorrt"`, `python -c "import torch; print(torch.cuda.is_available())"`.
- **Rule:** before planning any GPU work on a Jetson, run those four and paste the output into the plan.

## 2. The `nvidia-jetpack` metapackage is the wrong unit
- **Happened:** I started `apt-get install nvidia-jetpack` (CUDA toolkit with nvcc, samples, VPI, Nsight, OpenCV-CUDA,
  multimedia API; ~6 GB). The user asked for lean. I killed `apt-get` mid-download; dpkg kept unpacking ~110 packages.
- **Decision:** let dpkg finish, `dpkg --configure -a`, then `apt-mark auto` everything from that set, `apt-mark manual`
  the lean list, `apt-get autoremove --purge`. Final lean set (1.8 GB) that is sufficient for TensorRT inference through
  Ultralytics: `cuda-libraries-12-6`, `cuda-nvtx-12-6`, `cuda-cupti-12-6`, `libcudnn9-cuda-12`, `libnvinfer10`,
  `libnvinfer-plugin10`, `libnvonnxparsers10`, `libnvinfer-dispatch10`, `libnvinfer-lean10`, `python3-libnvinfer`,
  `nvidia-l4t-dla-compiler`.
- **Rules:** decide the package list before touching apt; never kill apt/dpkg during unpack; `--no-install-recommends`;
  pin versions (they are in the script).

## 3. `import tensorrt` failed after the lean install
- **Happened:** `ImportError: libnvdla_compiler.so` — the TensorRT Python module links the DLA compiler library, which lives
  in `nvidia-l4t-dla-compiler` (an L4T package the metapackage would have pulled). After installing it, `ldconfig` was
  needed before the loader found it.
- **Rule:** after any L4T/CUDA apt change run `sudo ldconfig`, then test the import from the exact interpreter you will use.

## 4. torch: where it comes from decides whether the GPU exists
- **Happened (three times):**
  1. The `uni` conda env had PyPI torch 2.3 for aarch64 → CPU-only.
  2. `pip install torch` with the Jetson index *and* `--extra-index-url pypi.org` resolved `torch==2.11.0+cu130` from PyPI:
     an SBSA/CUDA-13 build that installs fine and then says "NVIDIA driver on your system is too old".
  3. The Jetson wheel (`pypi.jetson-ai-lab.io/jp6/cu126`, torch 2.11.0 / torchvision 0.26.0, cp310) imports only with
     `libcudss.so.0`, which no apt package in the lean set provides.
- **Decision:** `pip install --no-deps --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch==2.11.0 torchvision==0.26.0`
  (no PyPI fallback, no dependency resolution), then `pip install --no-deps nvidia-cudss-cu12==0.8.0.10` for cuDSS and
  remove the CUDA 12.9 pip libraries it would otherwise drag in (`nvidia-cublas-cu12`, `nvidia-cuda-nvrtc-cu12`,
  `cuda-toolkit`) so the system CUDA 12.6 stays authoritative. Library paths go into
  `$CONDA_PREFIX/etc/conda/activate.d/cuda.sh` (`/usr/local/cuda-12.6/lib64` and the cuDSS wheel's `lib`).
- **Rules:** on Tegra, torch comes only from NVIDIA's Jetson index (or `dustynv` containers); PyPI and conda-forge aarch64
  builds are CPU or SBSA. Install with `--no-deps` and a single `--index-url`. Verify with `torch.cuda.is_available()`
  **and** a real op (`torch.ones(1000,1000,device="cuda") @ …`); `is_available()` alone passed on the wrong build.
  `ldd $SITE/torch/lib/libtorch_cuda.so | grep "not found"` lists what is still missing before you guess.

## 5. TensorRT's Python module lives outside conda
- **Happened:** `python3-libnvinfer` installs into `/usr/lib/python3.10/dist-packages`; the conda env could not see it.
- **Decision:** a one-line `.pth` file in the env's site-packages pointing at the system `dist-packages`. Works because the
  env's Python is also 3.10 (the binary module is version-specific). A conda env with another Python version would need
  the TensorRT wheel from the Jetson index instead.
- **Rule:** keep the env's Python at the system minor version on Jetson, or budget for sourcing TensorRT separately.

## 6. Ultralytics engine export needs extra packages
- **Happened:** `export(format="engine")` stopped at `No module named 'onnx'`. `onnx` and `onnxslim` are not pulled by
  `ultralytics`. Each FP16 engine took ~8 minutes to build on the Orin NX at 15 W.
- **Rules:** pin `onnx` + `onnxslim`; build engines on the target, in the background, with a log; keep `.pt` files to
  rebuild after any TensorRT or driver change. Engines are git-ignored; copy the `.pt` files over with `scp`.

## 7. Engine plan "different models of devices" and a one-time CUDA failure
- **Happened:** after a reboot TensorRT warned that the engine plan was built for a different device model and the very
  first inference failed (`cuTensor permutate execute failed` → CUDA stream-capture error). Subsequent runs worked.
- **Decision:** warm the detector up (two dummy frames) with one rebuild-and-retry **before** opening the RealSense; rebuild
  the engines on the current boot when convenient.
- **Rule:** never let the first real inference be the first inference; initialise CUDA/TensorRT before any other device
  library (librealsense/libusb) in the process.

## 8. The conda environment: clone, don't rebuild
- **Happened/decision:** pinocchio needs the conda-forge build for its CasADi bindings (the pip `pin` package lacks them)
  and rebuilding that on aarch64 is slow. The user had a working `uni` env. `conda create -n g1fetch --clone uni --offline`
  took a minute and kept `unitree_sdk2py`, `cyclonedds==0.10.2`, pinocchio 3.1, casadi 3.6.7, numpy 1.26 intact.
- **Rules:** clone an existing proven env offline when one exists; pin `numpy<2` (pinocchio ABI); on the x86 workstation
  import `pinocchio` before `torch` (conda libstdc++ must load first; `tests/conftest.py` does it).

## 9. pyrealsense2 on JetPack 6.2 just works from pip
- **Happened:** I expected to need the JetsonHacks kernel modules or a source build. `pip install pyrealsense2` (2.58.4,
  aarch64 wheel, RSUSB backend) delivered aligned RGB-D at 640×480/30 fps immediately — but only after stopping
  `teleimager-realsense-webrtc.service`, the V4L2 owner of the camera.
- **Rule:** check who owns `/dev/video*` (`systemctl list-units | grep -i realsense`) before debugging librealsense.

## 10. Process hygiene that cost time
- `pkill -f "<pattern>"` killed my own SSH shell because the pattern matched its command line → kill by PID or choose
  patterns that cannot match the caller.
- `sudo -v` in one shell does not carry into a `nohup` background job → `echo <pw> | sudo -S …` inside that job.
- A background job's working directory is not the interactive one → absolute paths for anything that writes files.
- I deleted a directory of freshly downloaded weights because I misjudged where a background job had written them →
  `ls` before `rm -rf`, always.
- Long installs: `nohup bash script > log 2>&1 &` on the target plus a waiter that greps the log for a DONE marker **and**
  for `error`; SSH sessions drop (the robot's Wi-Fi lease changed mid-session).
- Prove reproducibility by re-running the finished setup script in idempotent mode (`--no-apt`) on the same machine.

## 11. Numbers worth remembering (Orin NX 16 GB, 15 W mode, 4 cores online)
| step | time |
|---|---|
| lean apt set download + install | ~10 min, 1.8 GB |
| torch 2.11 Jetson wheel | ~300 MB |
| YOLOE-26s / YOLO26n TensorRT FP16 export | ~8 min each |
| both detectors on a camera frame, standalone | 51 ms |
| same inside the full robot process before the GIL fixes | 330–460 ms |
| same after the GIL fixes (`docs/05_deploy_learnings.md` B2) | ~90 ms |

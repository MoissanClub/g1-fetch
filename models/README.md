# models/

| file | what | produced by |
|---|---|---|
| `yoloe-26s-seg.pt` | Ultralytics YOLOE-26s base checkpoint (open-vocabulary, 22.9 MB) | auto-download |
| `mobileclip2_b.ts` | MobileCLIP text encoder used once to embed the prompts (242 MB) | auto-download, not needed on PC2 |
| `yoloe-26s-seg-fridge.pt` | YOLOE-26s with the four prompts baked in: `refrigerator`, `refrigerator door handle`, `soda can`, `person` | `scripts/export_detector.py --bake` |
| `yolo26n.pt` | COCO YOLO26n cross-check for `refrigerator` / `person` | auto-download |
| `*.engine` | TensorRT FP16 engines for the Orin NX (build on PC2, arch-specific) | `scripts/export_detector.py --engine` |

Nothing here was trained. Fine-tuning YOLOE on captured robot frames, if needed, runs on the Jetson
(`yolo train model=yoloe-26s-seg-fridge.pt data=... epochs=30 imgsz=640`) and must not use the workstation GPUs.

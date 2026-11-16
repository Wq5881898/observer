from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a small reproducible YOLO11 phone fine-tuning experiment")
    parser.add_argument("--data", type=Path, default=Path("data/training/yolo-phone-s96-runholdout/dataset.yaml"))
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics pretrained .pt model or checkpoint")
    parser.add_argument("--project", type=Path, default=Path("data/training/runs"))
    parser.add_argument("--name", default="yolo11n-phone-s96-baseline")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument('--cpu-threads', type=int, default=None,
                        help='limit CPU compute threads for this training process')
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--mosaic", type=float, default=0.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=3.0,
        help="Ultralytics warm-up duration; use 0 for very short incremental runs",
    )
    parser.add_argument(
        "--warmup-bias-lr",
        type=float,
        default=0.1,
        help="initial bias learning rate during warm-up",
    )
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="freeze this many leading model layers to preserve pretrained features",
    )
    parser.add_argument("--skip-val", action="store_true", help="skip Ultralytics validation during training; validate afterward with Observer")
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"dataset YAML not found: {args.data}")
    if args.epochs <= 0 or args.imgsz <= 0 or args.batch <= 0:
        raise SystemExit("epochs, imgsz, and batch must be positive")
    if args.freeze < 0:
        raise SystemExit("--freeze must be non-negative")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        raise SystemExit('--cpu-threads must be positive')
    if args.lr0 <= 0:
        raise SystemExit("--lr0 must be positive")
    if args.warmup_epochs < 0 or args.warmup_bias_lr < 0:
        raise SystemExit("warm-up values must be non-negative")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed in this Python environment. "
            "Install it in the isolated .yolo-train environment first."
        ) from exc

    model = YOLO(args.model)
    if args.cpu_threads is not None:
        # select_device resets torch thread count from this process-local value.
        from ultralytics.utils import torch_utils
        torch_utils.NUM_THREADS = args.cpu_threads
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=str(args.project.resolve()),
        name=args.name,
        pretrained=True,
        cache=False,
        exist_ok=False,
        mosaic=args.mosaic,
        mixup=args.mixup,
        fliplr=args.fliplr,
        lr0=args.lr0,
        warmup_epochs=args.warmup_epochs,
        warmup_bias_lr=args.warmup_bias_lr,
        freeze=args.freeze,
        optimizer="AdamW",
        patience=args.patience,
        close_mosaic=0,
        val=not args.skip_val,
        save_period=1 if args.skip_val else -1,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from datasets import create_test_dataset
from evaluator import Evaluator
from models import TETrack3D
from tracker import TETrack3DTracker
from utils import create_logger, load_config
from utils.checkpoint import load_model_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TETrack3D")
    parser.add_argument("config", help="Benchmark YAML configuration")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", help="Override dataset.root")
    parser.add_argument("--output-dir", default="./outputs/test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="bf16"
    )
    parser.add_argument("--max-tracklets", type=int)
    parser.add_argument(
        "--start-tracklet",
        type=int,
        default=0,
        help="Zero-based first tracklet, for deterministic evaluation sharding",
    )
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value):
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def resolve_amp_dtype(precision, device):
    if device.type != "cuda" or precision == "fp32":
        return None
    return torch.float16 if precision == "fp16" else torch.bfloat16


def main():
    args = parse_args()
    seed_everything(args.seed)
    config, config_path = load_config(args.config)
    if args.data_root:
        config.dataset.root = str(Path(args.data_root).expanduser().resolve())

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": str(config_path),
                "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
                "dataset_root": str(config.dataset.root),
                "seed": args.seed,
                "precision": args.precision,
                "start_tracklet": args.start_tracklet,
                "max_tracklets": args.max_tracklets,
            },
            handle,
            indent=2,
        )
    log = create_logger("TETrack3D-test", output_dir / "test.log")
    device = resolve_device(args.device)

    # State evolution supervision and OT alignment are training-only.
    model = TETrack3D(config.model, enable_training_modules=False)
    report = load_model_checkpoint(model, args.checkpoint, strict=True)
    model.to(device).eval()
    log.info(
        "Loaded %d inference tensors; ignored %d training-only or legacy tensors.",
        report.loaded,
        report.ignored,
    )

    dataset = create_test_dataset(
        config.dataset, log, debug=args.debug
    )
    tracker = TETrack3DTracker(
        model=model,
        config=config,
        device=device,
        amp_dtype=resolve_amp_dtype(args.precision, device),
    )
    evaluator = Evaluator(
        config=config,
        tracker=tracker,
        log=log,
        output_dir=output_dir,
        save_predictions=args.save_predictions,
    )
    evaluator.run(
        dataset,
        max_tracklets=args.max_tracklets,
        start_tracklet=args.start_tracklet,
    )


if __name__ == "__main__":
    main()

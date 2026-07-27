import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from datasets import create_training_datasets
from datasets.base_dataset import collate_tracklets
from training import TETrack3DTrainingModule
from utils import create_logger, load_config, save_config


DEFAULT_TRAINING_PRECISION = "bf16-mixed"


def parse_args():
    parser = argparse.ArgumentParser(description="Train TETrack3D")
    parser.add_argument("config", help="Experiment YAML configuration")
    parser.add_argument("--data-root", help="Override dataset.root")
    parser.add_argument(
        "--pretrained-backbone",
        help="RECON checkpoint used to initialize a fresh training run",
    )
    parser.add_argument("--resume", help="Resume a TETrack3D Lightning checkpoint")
    parser.add_argument("--output-dir", default="./outputs/train")
    parser.add_argument("--devices", default="0", help="GPU IDs, for example 0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def parse_devices(value):
    devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not devices:
        raise ValueError("At least one GPU ID is required.")
    return devices


def main():
    args = parse_args()
    config, _ = load_config(args.config)
    if args.data_root:
        config.dataset.root = str(Path(args.data_root).expanduser().resolve())
    configured_pretrained = getattr(config.model, "pretrained_backbone", None)
    pretrained_backbone = args.pretrained_backbone or configured_pretrained
    if not args.resume and not pretrained_backbone:
        raise ValueError(
            "A fresh run requires model.pretrained_backbone in the config (or "
            "--pretrained-backbone) because the paper uses a pretrained RECON "
            "encoder."
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")
    log = create_logger("TETrack3D-train", output_dir / "train.log")
    pl.seed_everything(args.seed, workers=True)

    train_dataset, validation_dataset = create_training_datasets(
        config.dataset, log, debug=args.debug
    )
    num_workers = 0 if args.debug else int(config.training.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=2 if args.debug else int(config.training.batch_size),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=False,
        num_workers=0 if args.debug else int(config.evaluation.num_workers),
        collate_fn=collate_tracklets,
        persistent_workers=(not args.debug and int(config.evaluation.num_workers) > 0),
    )

    module = TETrack3DTrainingModule(
        config=config,
        pretrained_backbone=None if args.resume else pretrained_backbone,
        text_logger=log,
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in module.model.parameters()
        if parameter.requires_grad
    )
    log.info("Trainable parameters: %.2f M", trainable_parameters / 1e6)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-{epoch:02d}-{val_precision:.3f}",
        monitor="val_precision",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    devices = parse_devices(args.devices)
    strategy = (
        DDPStrategy(find_unused_parameters=False) if len(devices) > 1 else "auto"
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        max_epochs=1 if args.debug else int(config.training.epochs),
        precision=DEFAULT_TRAINING_PRECISION,
        gradient_clip_val=float(config.training.gradient_clip),
        check_val_every_n_epoch=int(config.training.validation_interval),
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="epoch")],
        logger=CSVLogger(save_dir=output_dir, name="logs"),
        default_root_dir=output_dir,
        log_every_n_steps=1 if args.debug else 20,
        limit_train_batches=2 if args.debug else 1.0,
        limit_val_batches=2 if args.debug else 1.0,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        module,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path=args.resume,
    )


if __name__ == "__main__":
    main()

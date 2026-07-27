# TETrack3D

Official training and evaluation code for **State Evolution Awareness for
Category-agnostic 3D Point Cloud Tracking**.

## Structure

```text
configs/       Official KITTI, nuScenes, and Waymo benchmark configurations
datasets/      Training and benchmark dataset adapters
models/        RECON backbone, localization head, and state evolution modules
training/      TETrack3D objectives and Lightning training module
train.py       Training entry point
test.py        Evaluation entry point
```

The repository intentionally contains no datasets, RECON initialization
weights, or trained TETrack3D checkpoints. The official configs reference the
external RECON initialization file through `model.pretrained_backbone`; update
that path when moving the code to another machine.

## Environment

The exported environment used for training and evaluation is provided in
`environment.yml`. It contains Python 3.10, PyTorch 2.1.2, CUDA 11.8,
PyTorch3D 0.7.8, and Mamba-SSM 1.2.0.post1.

```bash
conda env create -f environment.yml
conda activate 3dtracktorch2_224_mamba_test
```

The target machine needs an NVIDIA driver compatible with CUDA 11.8. If Conda
cannot reuse the exported PyTorch3D or Mamba packages on a different GPU
architecture, reinstall those two packages against the PyTorch and CUDA
versions above.

## Data

Dataset paths are read from `dataset.root` in each config and can be overridden
with `--data-root`. The expected KITTI tracking layout is:

```text
KITTI/training/
├── calib/
├── label_02/
└── velodyne/
```

For nuScenes, pass the directory containing both the metadata and point-cloud
folders:

```text
nuScenes/
├── v1.0-trainval/
├── samples/
├── sweeps/
└── maps/
```


## Train

Training **requires a pretrained RECON encoder**. Before starting a run, either
set `model.pretrained_backbone` in the selected YAML config to a valid RECON
checkpoint or provide the checkpoint with `--pretrained-backbone`. Do not train
TETrack3D without this initialization.

```bash
python train.py configs/tetrack3d_kitti.yaml \
  --data-root /path/to/KITTI/training \
  --pretrained-backbone /path/to/modelnet8k_94_28.pth \
  --devices 0,1,2,3 \
  --output-dir outputs/kitti
```

For nuScenes, select `configs/tetrack3d_nuscenes.yaml` and pass the nuScenes
root shown above.

The command-line option overrides `model.pretrained_backbone` in the YAML
config. RECON initialization loads the backbone alone; it does not initialize
the localization or state-evolution modules.

Resume an interrupted TETrack3D run with `--resume /path/to/last.ckpt`.
`--debug` runs two training and validation batches for a quick wiring check.

## Evaluate

```bash
python test.py configs/tetrack3d_kitti.yaml \
  --data-root /path/to/KITTI/training \
  --checkpoint /path/to/tetrack3d.ckpt \
  --device cuda:0 \
  --output-dir outputs/kitti_test
```

The checkpoint loader strictly validates all inference weights.

Use `configs/tetrack3d_nuscenes.yaml` for nuScenes. The Waymo result in the
paper uses the KITTI-trained checkpoint with `configs/tetrack3d_waymo.yaml`.
The evaluator writes `test_summary.json` and can optionally export predicted
boxes with `--save-predictions`.

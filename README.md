# TETrack3D: State Evolution Awareness for Category-agnostic 3D Point Cloud Tracking

_Official PyTorch implementation · NeurIPS 2026_

---

This repository contains the training and evaluation code for **TETrack3D**, the method introduced in _State Evolution Awareness for Category-agnostic 3D Point Cloud Tracking_ (NeurIPS 2026). The code supports category-agnostic training and evaluation on KITTI and nuScenes, as well as direct KITTI-to-Waymo evaluation.

## 📋 Repository structure

```text
.
├── configs/          # KITTI, nuScenes, and Waymo experiment configurations
├── datasets/         # Dataset adapters and point-cloud preprocessing
├── models/           # RECON backbone, localization head, and state evolution
├── training/         # Losses and PyTorch Lightning training module
├── utils/            # Configuration, logging, metrics, and checkpoint utilities
├── weights/          # Released TETrack3D checkpoints
├── environment.yml   # Reproducible Conda environment
├── train.py          # Training entry point
└── test.py           # Evaluation entry point
```

Datasets and the pretrained RECON initialization weights are not included. Configure their paths before training or evaluation.

## 🔧 Environment setup

Create the Conda environment from the provided [`environment.yml`](environment.yml). This is the required environment specification for this repository; no separate `pip install` step is needed.

```bash
cd /share/hxt/tracking/some_try/3d_my/eccv/ours
conda env create -f /share/hxt/tracking/some_try/3d_my/eccv/ours/environment.yml
conda activate 3dtracktorch2_224_mamba_test
```

The environment includes Python 3.10, PyTorch 2.1.2, CUDA 11.8, PyTorch3D 0.7.8, PyTorch Lightning 2.1.0, and Mamba-SSM 1.2.0.post1. An NVIDIA driver compatible with CUDA 11.8 is required.

To update an existing environment from the same file, run:

```bash
conda env update \
  -n 3dtracktorch2_224_mamba_test \
  -f /share/hxt/tracking/some_try/3d_my/eccv/ours/environment.yml \
  --prune
```

## 💾 Dataset preparation

Dataset roots can be set in the corresponding file under `configs/` or overridden at runtime with `--data-root`.

### KITTI

Set the data root to the KITTI `training` directory:

```text
KITTI/training/
├── calib/
├── label_02/
└── velodyne/
```

### nuScenes

Set the data root to the directory containing both the metadata and point-cloud folders:

```text
nuScenes/
├── v1.0-trainval/
├── samples/
├── sweeps/
└── maps/
```

### Waymo

The Waymo adapter expects the preprocessed tracking benchmark layout below:

```text
Waymo/
├── benchmark/validation/vehicle/
│   ├── bench_list.json
│   ├── easy.json
│   ├── medium.json
│   └── hard.json
├── gt_info/
└── pc/raw_pc/
```

Waymo is evaluation-only. The reported Waymo setting directly evaluates the KITTI-trained model without Waymo fine-tuning.

## 📦 Checkpoints

Download the released TETrack3D checkpoints from [Google Drive](https://drive.google.com/drive/folders/1MQbxJXfzGvKflySuFNo34vgXj8mOjwPG?usp=drive_link), then place them in the `weights/` directory:

| Dataset | Checkpoint | Usage |
| ------- | ---------- | ----- |
| **KITTI** | `weights/tetrack3d_kitti.ckpt` | KITTI evaluation |
| **nuScenes** | `weights/tetrack3d_nuscenes.ckpt` | nuScenes evaluation |
| **Waymo** | `weights/tetrack3d_waymo.ckpt` | KITTI-to-Waymo evaluation |

The Waymo checkpoint contains the same model weights as the KITTI checkpoint because the Waymo experiment uses direct cross-dataset evaluation.

Fresh training additionally requires a pretrained RECON encoder. Set `model.pretrained_backbone` in the selected configuration or pass its path with `--pretrained-backbone`.

## 📊 Evaluation

Run all commands from the repository root after activating the Conda environment.

### KITTI

```bash
python test.py configs/tetrack3d_kitti.yaml \
  --data-root /path/to/KITTI/training \
  --checkpoint weights/tetrack3d_kitti.ckpt \
  --device cuda:0 \
  --output-dir outputs/kitti_test
```

### nuScenes

```bash
python test.py configs/tetrack3d_nuscenes.yaml \
  --data-root /path/to/nuScenes \
  --checkpoint weights/tetrack3d_nuscenes.ckpt \
  --device cuda:0 \
  --output-dir outputs/nuscenes_test
```

### Waymo

```bash
python test.py configs/tetrack3d_waymo.yaml \
  --data-root /path/to/Waymo \
  --checkpoint weights/tetrack3d_waymo.ckpt \
  --device cuda:0 \
  --output-dir outputs/waymo_test
```

Evaluation writes the aggregate metrics to `test_summary.json` and the run metadata to `evaluation.json`. Add `--save-predictions` to export predicted boxes. For a quick pipeline check, add `--debug --max-tracklets 2`.

## ⚙️ Training

Training is supported on KITTI and nuScenes. A fresh run must initialize the backbone from a pretrained RECON checkpoint.

### KITTI

```bash
python train.py configs/tetrack3d_kitti.yaml \
  --data-root /path/to/KITTI/training \
  --pretrained-backbone /path/to/modelnet8k_94_28.pth \
  --devices 0,1,2,3 \
  --output-dir outputs/kitti_train
```

### nuScenes

```bash
python train.py configs/tetrack3d_nuscenes.yaml \
  --data-root /path/to/nuScenes \
  --pretrained-backbone /path/to/modelnet8k_94_28.pth \
  --devices 0,1,2,3 \
  --output-dir outputs/nuscenes_train
```

The command-line `--pretrained-backbone` value overrides `model.pretrained_backbone` in the YAML configuration. It initializes only the RECON backbone, not the localization or state-evolution modules.

Resume an interrupted run with:

```bash
python train.py configs/tetrack3d_kitti.yaml \
  --data-root /path/to/KITTI/training \
  --resume /path/to/last.ckpt \
  --devices 0,1,2,3 \
  --output-dir outputs/kitti_train
```

Add `--debug` to limit training and validation to two batches for a quick wiring check.

## 📚 Citation

If this repository is useful for your research, please cite the NeurIPS 2026 paper:

```text
State Evolution Awareness for Category-agnostic 3D Point Cloud Tracking.
Advances in Neural Information Processing Systems (NeurIPS), 2026.
```

The complete BibTeX entry will be added with the public paper metadata.

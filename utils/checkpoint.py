from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class LoadReport:
    loaded: int
    ignored: int


_LEGACY_PREFIXES = {
    "backbone_net.": "backbone.",
    "loc_net.": "localization_head.",
    "state_predictor.": "temporal_predictor.",
}


def _model_key(raw_key):
    key = raw_key
    for prefix in ("module.model.", "model.", "module."):
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    for old_prefix, new_prefix in _LEGACY_PREFIXES.items():
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]
    return key


def load_model_checkpoint(model, checkpoint_path, strict=True):
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {path}")

    model_state = model.state_dict()
    compatible = {}
    ignored = 0
    for raw_key, value in state_dict.items():
        if not torch.is_tensor(value):
            ignored += 1
            continue
        key = _model_key(raw_key)
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            ignored += 1

    missing = sorted(set(model_state) - set(compatible))
    if strict and missing:
        preview = "\n  ".join(missing[:20])
        raise RuntimeError(
            f"Checkpoint is missing {len(missing)} required tensors:\n  {preview}"
        )
    model.load_state_dict(compatible, strict=False)
    return LoadReport(loaded=len(compatible), ignored=ignored)

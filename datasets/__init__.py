from addict import Dict

from .kitti import KITTITrackingDataset
from .waymo import WaymoTrackingDataset


def _dataset_config(config, debug=False):
    adapted = Dict(config.to_dict())
    adapted.debug = bool(debug)
    adapted.data_root_dir = str(config.root)
    adapted.category_name = str(config.category)
    adapted.frame_npts = int(config.num_points)
    adapted.num_smp_frames_per_tracklet = int(config.sequence_length)
    adapted.max_frame_dis = int(config.max_frame_distance)
    adapted.train_cfg = adapted.train
    adapted.eval_cfg = adapted.test
    return adapted


def _dataset_class(name):
    name = str(name).lower()
    if name == "kitti":
        return KITTITrackingDataset
    if name == "nuscenes":
        from .nuscenes import NuScenesTrackingDataset

        return NuScenesTrackingDataset
    if name == "waymo":
        return WaymoTrackingDataset
    raise ValueError(f"Unsupported dataset: {name}")


def create_dataset(config, split, log, debug=False):
    dataset_config = _dataset_config(config, debug=debug)
    dataset = _dataset_class(config.name)(split, dataset_config, log)
    return dataset.get_dataset()


def create_training_datasets(config, log, debug=False):
    if str(config.name).lower() == "waymo":
        raise ValueError(
            "The paper evaluates Waymo using the KITTI-trained model; "
            "the Waymo benchmark adapter is test-only."
        )
    return (
        create_dataset(config, "train", log, debug=debug),
        create_dataset(config, "val", log, debug=debug),
    )


def create_test_dataset(config, log, debug=False):
    return create_dataset(config, "test", log, debug=debug)


__all__ = ["create_test_dataset", "create_training_datasets"]

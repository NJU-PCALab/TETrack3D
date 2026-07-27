import numpy as np
import torch
import torchmetrics.utilities.data
from shapely.geometry import Polygon
from torchmetrics import Metric


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def estimateAccuracy(box_a, box_b, dim=3, up_axis=(0, -1, 0)):
    if dim == 3:
        return np.linalg.norm(box_a.center - box_b.center, ord=2)
    elif dim == 2:
        up_axis = np.array(up_axis)
        return np.linalg.norm(
            box_a.center[up_axis == 0] - box_b.center[up_axis == 0], ord=2)


def fromBoxToPoly(box, up_axis=(0, -1, 0)):
    """

    :param box:
    :param up_axis: the up axis must contain only one non-zero component
    :return:
    """
    if up_axis[1] != 0:
        return Polygon(tuple(box.corners()[[0, 2]].T[[0, 1, 5, 4]]))
    elif up_axis[2] != 0:
        return Polygon(tuple(box.bottom_corners().T))


def estimateOverlap(box_a, box_b, dim=2, up_axis=(0, -1, 0)):
    # if box_a == box_b:
    #     return 1.0

    Poly_anno = fromBoxToPoly(box_a, up_axis)
    Poly_subm = fromBoxToPoly(box_b, up_axis)

    box_inter = Poly_anno.intersection(Poly_subm)
    box_union = Poly_anno.union(Poly_subm)
    if dim == 2:
        return box_inter.area / box_union.area

    else:
        up_axis = np.array(up_axis)
        up_max = min(box_a.center[up_axis != 0], box_b.center[up_axis != 0])
        up_min = max(box_a.center[up_axis != 0] - box_a.wlh[2],
                     box_b.center[up_axis != 0] - box_b.wlh[2])
        inter_vol = box_inter.area * max(0, up_max[0] - up_min[0])
        anno_vol = box_a.wlh[0] * box_a.wlh[1] * box_a.wlh[2]
        subm_vol = box_b.wlh[0] * box_b.wlh[1] * box_b.wlh[2]

        overlap = inter_vol * 1.0 / (anno_vol + subm_vol - inter_vol)
        return overlap


def fromWaymoBoxToPoly(box):
    return Polygon(tuple(box.corners()[[0, 1]].T[[0, 1, 5, 4]]))


def estimateWaymoOverlap(box_a, box_b, dim=2):

    Poly_anno = fromWaymoBoxToPoly(box_a)
    Poly_subm = fromWaymoBoxToPoly(box_b)

    box_inter = Poly_anno.intersection(Poly_subm)
    box_union = Poly_anno.union(Poly_subm)
    if dim == 2:
        return box_inter.area / box_union.area
    else:
        zmax = min(box_a.center[2], box_b.center[2])
        zmin = max(box_a.center[2] - box_a.wlh[2],
                   box_b.center[2] - box_b.wlh[2])
        inter_vol = box_inter.area * max(0, zmax - zmin)
        anno_vol = box_a.wlh[0] * box_a.wlh[1] * box_a.wlh[2]
        subm_vol = box_b.wlh[0] * box_b.wlh[1] * box_b.wlh[2]
        overlap = inter_vol * 1.0 / (anno_vol + subm_vol - inter_vol)
    return overlap


class TorchPrecision(Metric):
    """Computes and stores the Precision using torchMetrics"""

    def __init__(self, n=21, max_accuracy=2, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.max_accuracy = max_accuracy
        self.Xaxis = torch.linspace(0, self.max_accuracy, steps=n)
        self.add_state("accuracies", default=[])

    def value(self, accs):
        prec = [
            torch.sum((accs <= thres).float()) / len(accs)
            for thres in self.Xaxis
        ]
        return torch.tensor(prec)

    def update(self, val):
        self.accuracies.append(val)

    def compute(self):
        accs = torchmetrics.utilities.data.dim_zero_cat(self.accuracies)
        if accs.numel() == 0:
            return 0.0
        return torch.trapz(self.value(accs), x=self.Xaxis * 100 / self.max_accuracy)


class TorchSuccess(Metric):
    """Computes and stores the Success using torchMetrics"""

    def __init__(self, n=21, max_overlap=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.max_overlap = max_overlap
        self.Xaxis = torch.linspace(0, self.max_overlap, steps=n)
        self.add_state("overlaps", default=[])

    def value(self, overlaps):
        succ = [
            torch.sum((overlaps >= thres).float()) / len(overlaps)
            for thres in self.Xaxis
        ]
        return torch.tensor(succ)

    def compute(self):
        overlaps = torchmetrics.utilities.data.dim_zero_cat(self.overlaps)

        if overlaps.numel() == 0:
            return 0
        return torch.tensor(np.trapz(self.value(overlaps), x=self.Xaxis) * 100 / self.max_overlap)

    def update(self, val):
        self.overlaps.append(val)


class TorchRuntime(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("sum_runtime", default=torch.tensor(0.0, dtype=torch.float),
                       dist_reduce_fx='sum')
        self.add_state("num_runs", default=torch.tensor(0, dtype=torch.int),
                       dist_reduce_fx='sum')

    def update(self, runtime, n_runs):
        self.sum_runtime += runtime
        self.num_runs += n_runs

    def compute(self):
        return self.sum_runtime / self.num_runs


class TorchNumFrames(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("n_frames", default=torch.tensor(0.0, dtype=torch.float),
                       dist_reduce_fx='sum')

    def update(self, n_frames):
        self.n_frames += n_frames

    def compute(self):
        return self.n_frames

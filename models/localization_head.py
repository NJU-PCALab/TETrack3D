
import numpy as np
import torch
from torch import nn
from pytorch3d.ops import sample_farthest_points, knn_points, knn_gather

from . import layers


class EdgeAggregator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mlps = list(cfg.mlps)
        if not cfg.use_xyz:
            raise ValueError("Edge aggregation requires XYZ coordinates.")
        mlps[0] += 6
        pre_mlps = cfg.pre_mlps
        self.pre_shared_mlp = (
            layers.Seq(pre_mlps[0])
            .conv1d(pre_mlps[1], bn=True)
            .conv1d(pre_mlps[2], activation=None)
        )
        self.shared_mlp = layers.SharedMLP(mlps, bn=True)
        self.cfg = cfg

    def get_graph_feature(self, new_xyz, xyz, feat, k):
        feat = feat.permute(0, 2, 1).contiguous() if feat is not None else None
        # b,n,c

        _, knn_idx, knn_xyz = knn_points(
            new_xyz, xyz, K=k, return_nn=True)
        # b, n1, k, 3

        knn_feat = knn_gather(feat, knn_idx)
        # b,n1,k,c

        xyz_tiled = new_xyz.unsqueeze(-2).repeat(1, 1, k, 1)
        edge_feat = torch.cat([knn_xyz-xyz_tiled, xyz_tiled, knn_feat], dim=-1)
        return edge_feat.permute(0, 3, 1, 2).contiguous()

    def forward(self, new_xyz, xyz, feat):
        feat = self.pre_shared_mlp(feat)
        edge_feat = self.get_graph_feature(
            new_xyz, xyz, feat, self.cfg.nsample)
        new_feat = self.shared_mlp(edge_feat)
        new_feat = new_feat.max(dim=-1, keepdim=False)[0]
        return new_feat


class LocalizationHead(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc_mask = (
            layers.Seq(cfg.feat_dim)
            .conv1d(cfg.feat_dim, bn=True)
            .conv1d(cfg.feat_dim, bn=True)
            .conv1d(1, activation=None))
        self.fc_center = (
            layers.Seq(3 + cfg.feat_dim)
            .conv1d(cfg.feat_dim, bn=True)
            .conv1d(cfg.feat_dim, bn=True)
            .conv1d(3 + 1 + cfg.feat_dim, activation=None))
        self.center_emb = (
            layers.Seq(1)
            .conv1d(cfg.feat_dim, activation=None)
        )
        self.edge_aggr = EdgeAggregator(cfg.edge_aggr)
        self.aggr_conv = (
            layers.Seq(cfg.feat_dim * cfg.n_smp_x *
                         cfg.n_smp_y * cfg.n_smp_z)
            .conv1d(cfg.feat_dim, bn=True)
        )
        self.prototype_points = self._build_prototype_points(
            cfg.n_smp_x, cfg.n_smp_y, cfg.n_smp_z
        )

        self.fc_proposal = (
            layers.Seq(cfg.feat_dim)
            .conv1d(cfg.feat_dim, bn=True)
            .conv1d(cfg.feat_dim, bn=True)
            .conv1d(3 + 1 + 1, activation=None))

    def _build_prototype_points(self, n_smp_x, n_smp_y, n_smp_z):
        samples = []
        if self.cfg.sample_method == 'shrink':
            for i in range(0, n_smp_x):
                for j in range(0, n_smp_y):
                    for k in range(0, n_smp_z):
                        samples.append((1.0*(i+0.5)/n_smp_x, 1.0*(j+0.5) /
                                        n_smp_y, 1.0*(k+0.5)/n_smp_z))
        elif self.cfg.sample_method == 'vanilla':
            for i in range(0, n_smp_x):
                for j in range(0, n_smp_y):
                    for k in range(0, n_smp_z):
                        samples.append((1.0*i/(n_smp_x-1), 1.0*j /
                                        (n_smp_y-1), 1.0*k/(n_smp_z-1)))
        else:
            raise ValueError(f"Unsupported proposal sampling: {self.cfg.sample_method}")
        # Keep the proposal grid in NumPy's default float64 representation and
        # cast it directly to the current feature dtype in sample_points().
        # Pre-quantizing the grid to float32 changes BF16 rounding at proposal
        # boundaries and can make farthest-point sampling select another box.
        samples = np.asarray(samples)
        samples -= 0.5
        return samples

    def sample_points(self, c_xyz, lwh):

        bs = c_xyz.shape[0]
        n_proposals = c_xyz.shape[1]
        points = torch.tensor(
            self.prototype_points,
            device=c_xyz.device,
            dtype=c_xyz.dtype,
        )

        # ns, 3
        n_samples = points.shape[0]
        points = points.unsqueeze(0).repeat(
            bs, 1, 1) * lwh.unsqueeze(1).repeat(1, n_samples, 1)
        # b, ns, 3
        sample_xyz = c_xyz.unsqueeze(2).repeat(
            1, 1, n_samples, 1) + points.unsqueeze(1).repeat(1, n_proposals, 1, 1)
        return sample_xyz

    def forward(self, inputs):
        xyz = inputs["xyz"]
        features = inputs["features"] + inputs["mask_features"]
        box_size = inputs["box_size"]

        mask_pred = self.fc_mask(features).squeeze(1)

        feat_xyz = torch.cat(
            (features, xyz.permute(0, 2, 1).contiguous()), dim=1)
        offset = self.fc_center(feat_xyz)
        offset_feat, offset_center_pred, objectness_pred = torch.split(
            offset, [self.cfg.feat_dim, 3, 1], dim=1)

        features = features + offset_feat
        offset_center_pred = offset_center_pred.permute(0,2,1).contiguous()
        center_pred = offset_center_pred + xyz
        objectness_pred = objectness_pred.squeeze(1)

        proposal_features = torch.cat(
            (mask_pred.sigmoid().unsqueeze(1), features), dim=1
        )

        proposal_xyz, fps_index = sample_farthest_points(
            center_pred, K=self.cfg.n_proposals
        )
        objectness_score = torch.gather(
            objectness_pred.sigmoid(), dim=-1, index=fps_index
        )
        sample_xyz = self.sample_points(proposal_xyz, box_size)

        bs = sample_xyz.shape[0]
        n_rois = sample_xyz.shape[1]
        n_roi_samples = sample_xyz.shape[2]

        sample_xyz = sample_xyz.reshape(bs, n_rois * n_roi_samples, 3)
        proposal_sample_feat = self.edge_aggr(
            sample_xyz, xyz, proposal_features
        )
        proposal_sample_feat = proposal_sample_feat.reshape(
            bs, -1, n_rois, n_roi_samples)
        proposal_feat = self.aggr_conv(proposal_sample_feat.permute(
            0, 1, 3, 2).reshape(bs, -1, n_rois))

        ce = objectness_score.unsqueeze(1)
        ce = self.center_emb(ce)

        proposal_offsets = self.fc_proposal(
            proposal_feat+ce).transpose(1, 2).contiguous()
        estimation_boxes = torch.cat(
            (proposal_offsets[:, :, 0:3] +
             proposal_xyz, proposal_offsets[:, :, 3:5]),
            dim=-1)

        return {
            "offset_center": offset_center_pred,
            "mask_logits": mask_pred,
            "center_objectness_logits": objectness_pred,
            "center": center_pred,
            "boxes": estimation_boxes,
            "proposal_centers": proposal_xyz,
        }

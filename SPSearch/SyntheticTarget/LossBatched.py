from typing import Union
import torch
from pytorch3d.loss.chamfer import _validate_chamfer_reduction_inputs, _handle_pointcloud_input
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops import sample_points_from_meshes

from kornia.geometry import depth_to_normals
import torchvision
# from kornia.geometry import depth_to_3d_v2
# import kornia.core as kornia_ops
# from kornia.filters.sobel import spatial_gradient


from PytorchGeoNodes.Pytorch3DRenderer.Torch3DRenderer import Torch3DRenderer


def one_way_chamfer_distance(
    x,
    y,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
    norm: int = 2,
):
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, None, None)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, None, None)

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]


    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)

    # clamp to avoid wrong correspondence
    cham_x = torch.clamp_max(cham_x, max=0.2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    # cham_y = cham_y.sum(1)  # (N,)

    if point_reduction == "mean":
        x_lengths_clamped = x_lengths.clamp(min=1)
        cham_x /= x_lengths_clamped

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        if batch_reduction == "mean":
            div = max(N, 1)
            cham_x /= div

    cham_dist = cham_x
    return cham_dist

class LossBatched(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.torch_renderer = Torch3DRenderer()

    def forward(self, obj_mesh, scene_dict):
        device = obj_mesh.device

        # scene_dict contains:
        # {
        #     'scene_pcd',
        #     'scene_mask',
        #     'scene_depth',
        #     'renderer'
        # }

        depth_gt = scene_dict['scene_depth']

        # depth_gt = depth_gt.float().to(device)
        # mask_depth_valid_sensor = depth_gt != 0.0
        mask_depth_valid_sensor = torch.ones_like(depth_gt).to(device)
        mask_depth_valid_sensor = mask_depth_valid_sensor.float()

        mask_gt = scene_dict['scene_mask'].float()

        renderer = scene_dict['renderer']

        n_views = renderer.rasterizer.cameras.T.shape[0]

        obj_mesh_extend = obj_mesh.extend(n_views)

        rendered_mesh, zbuf = renderer(obj_mesh_extend)
        rendered_mesh = rendered_mesh[..., 1]

        mesh_mask = rendered_mesh != 0

        depth_pred = zbuf[..., 0]

        depth_pred[depth_pred < 0.0] = torch.max(depth_gt)

        mask_pred = mesh_mask.float()
        mask_depth_bg = torch.logical_not(mesh_mask).float()

        depth_final = depth_pred * mask_pred + depth_gt * mask_depth_bg * (1 - mask_gt)

        loss_dict = self.calculate_render_loss(rendered_mesh, mask_gt,
                                               depth_final,
                                               depth_gt,
                                               mask_depth_valid_sensor,
                                               0)

        total_loss_randc_loss = loss_dict['total_loss']
        total_loss_randc_loss = total_loss_randc_loss.mean()

        target_obj_pcd = scene_dict['scene_pcd']

        mesh_pcd = (
            sample_points_from_meshes(obj_mesh, num_samples=10000))

        # calculate chamfer loss
        chamfer_loss = one_way_chamfer_distance(target_obj_pcd, mesh_pcd)

        total_loss = total_loss_randc_loss + 10 * chamfer_loss

        return total_loss

    def calculate_render_loss(self, mask_pred, mask_gt, depth_final, depth_sensor,
                              mask_depth_valid_sensor, mask_depth_valid_render_pred):
        # Simplified loss while ignoring "special cases"

        gauss_blur = torchvision.transforms.GaussianBlur(3, sigma=5)
        mask_gt = gauss_blur(mask_gt)
        mask_pred = gauss_blur(mask_pred)

        with torch.no_grad():
            loss_sil = torch.mean(torch.abs(mask_pred - mask_gt), dim=(1, 2))

        loss_sensor = torch.abs(mask_depth_valid_sensor *
                                (depth_sensor - depth_final))


        loss_sensor = loss_sensor.mean(dim=(1, 2))

        K = torch.eye(3)[None]
        K = K.expand(depth_sensor.shape[0], -1, -1).to(depth_sensor.device)

        normals_sensor = depth_to_normals(depth_sensor[:, None], K, normalize_points=True)
        normals_final = depth_to_normals(depth_final[:, None], K, normalize_points=True)

        loss_normals = (
            mask_depth_valid_sensor[:, None] * (normals_sensor - normals_final)).abs().mean(dim=(1, 2, 3))

        total_loss = loss_sensor + loss_normals

        loss_dict = {
            'total_loss': total_loss,
        }

        return loss_dict

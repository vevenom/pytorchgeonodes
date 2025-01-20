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
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    clamp_max=0.2,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
    norm: int = 2,
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
          :param clamp_max:
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]


    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)

    # clamp to avoid wrong correspondence
    cham_x = torch.clamp_max(cham_x, max=0.2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0


    if weights is not None:
        cham_x *= weights.view(N, 1)

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
            div = weights.sum() if weights is not None else max(N, 1)
            cham_x /= div

    cham_dist = cham_x
    return cham_dist

class LossBatchedSlow(torch.nn.Module):
    def __init__(self, settings):
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

        depth_gt = scene_dict['scene_depth'].clone()

        mask_gt = scene_dict['scene_mask'].float()

        renderer = scene_dict['renderer']

        n_views = renderer.rasterizer.cameras.T.shape[0]

        obj_mesh_extend = obj_mesh.extend(n_views)

        rendered_mesh, zbuf = renderer(obj_mesh_extend)
        rendered_mesh = rendered_mesh[..., 1]

        depth_max = depth_gt[mask_gt > 0].max()
        depth_gt[mask_gt < 1] = depth_max

        mesh_mask = rendered_mesh != 0

        depth_pred = zbuf[..., 0]

        depth_pred[depth_pred < 0.0] = depth_max

        mask_pred = mesh_mask.float()
        mask_depth_bg = torch.logical_not(mesh_mask).float()

        loss_dict = self.calculate_loss(rendered_mesh, mask_gt,
                                        depth_pred,
                                        depth_gt,
                                        0)


        total_loss_randc_loss = loss_dict['total_loss']
        total_loss_randc_loss = total_loss_randc_loss.mean()

        target_obj_pcd = scene_dict['scene_pcd']

        mesh_pcd = (
            sample_points_from_meshes(obj_mesh, num_samples=10000))

        # calculate chamfer loss
        chamfer_loss = one_way_chamfer_distance(target_obj_pcd, mesh_pcd)

        #
        # print('Chamfer loss: {}'.format(chamfer_loss * 10))
        # print('Render loss: {}'.format(total_loss_randc_loss))

        total_loss = total_loss_randc_loss + 10 * chamfer_loss

        return total_loss


    def calculate_loss(self, mask_pred, mask_gt, depth_pred, depth_sensor,
                       mask_depth_valid_render_pred):
        # Simplified loss while ignoring "special cases"

        # loss_sil = 1 - (torch.sum((mask_pred * mask_gt), dim=(1, 2)) /
        #                 torch.sum((mask_pred + mask_gt - (mask_pred * mask_gt)), dim=(1, 2)))

        # gauss_blur = torchvision.transforms.GaussianBlur(3, sigma=5)
        # mask_gt = gauss_blur(mask_gt)
        # mask_pred = gauss_blur(mask_pred)
        # with torch.no_grad():
        #     loss_sil = torch.mean(torch.abs(mask_pred - mask_gt), dim=(1, 2))
        # loss_sil = torch.mean(torch.abs(mask_pred - mask_gt), dim=(1, 2))

        # import matplotlib.pyplot as plt
        #
        # plt.subplot(141)
        # plt.imshow(mask_pred[0].detach().cpu().numpy())
        # plt.subplot(142)
        # plt.imshow(mask_gt[0].detach().cpu().numpy())
        # plt.subplot(143)
        # plt.imshow(depth_pred[0].detach().cpu().numpy())
        # plt.subplot(144)
        # plt.imshow(depth_sensor[0].detach().cpu().numpy())
        #
        #
        # plt.show()

        # loss_depth = (
        #     ((depth_gt - depth_final) * mask_depth_valid_render_gt * mask_depth_valid_render_pred).abs().mean(
        #     dim=(1, 2)))
        # loss_sensor = (
        #     ((depth_sensor - depth_final) * mask_depth_valid_sensor * mask_depth_valid_render_pred).abs().mean(
        #         dim=(1, 2)))

        loss_sensor = torch.abs((depth_sensor - depth_pred))

        # loss_sensor = (
        #         mask_depth_valid_sensor *
        #         (depth_sensor - depth_final)).abs().mean(dim=(1, 2))

        # clamp loss to 0.05 and normalize
        # loss_sensor = torch.clamp(loss_sensor, max=0.05) / 0.05

        loss_sensor = loss_sensor.mean(dim=(1, 2))

        # loss_sensor = (
        #         mask_depth_valid_sensor *
        #         (depth_sensor - depth_final) ** 2).mean(dim=(1, 2))

        K = torch.eye(3)[None]
        K = K.expand(depth_sensor.shape[0], -1, -1).to(depth_sensor.device)
        # print(K.shape)
        # assert False

        # # this is depth_to_normals from kornia with depth_to_3d_v2 to ignore the current deprecation warning
        # def depth_to_normals(depth, intrinsics, normalize_points=False):
        #
        #     xyz = depth_to_3d_v2(depth, intrinsics, normalize_points)  # Bx3xHxW
        #     # compute the pointcloud spatial gradients
        #     gradients = spatial_gradient(xyz)  # Bx3x2xHxW
        #     # compute normals
        #     a, b = gradients[:, :, 0], gradients[:, :, 1]  # Bx3xHxW
        #     normals = torch.cross(a, b, dim=1)  # Bx3xHxW
        #     normals = kornia_ops.normalize(normals, dim=1, p=2)
        #
        #     return normals

        normals_sensor = depth_to_normals(depth_sensor[:, None], K, normalize_points=True)
        normals_final = depth_to_normals(depth_pred[:, None], K, normalize_points=True)

        # from matplotlib import pyplot as plt
        #
        # # display normals but swap C,H,W to H,W,C
        # plt.subplot(121)
        # plt.imshow(normals_sensor[10].permute(1, 2, 0).detach().cpu().numpy())
        # plt.subplot(122)
        # plt.imshow(normals_final[10].permute(1, 2, 0).detach().cpu().numpy())
        # plt.show()

        loss_normals = ((normals_sensor - normals_final)).abs().mean(dim=(1, 2, 3))

        # loss_normals = (
        #         mask_depth_valid_sensor[:, None] * (normals_sensor - normals_final) ** 2).mean(dim=(1, 2, 3))


        # total_loss = loss_sil + loss_depth + loss_sensor
        # total_loss = loss_sil + loss_sensor
        # total_loss = loss_sil
        total_loss = loss_sensor + loss_normals
        # total_loss = loss_sensor
        # total_loss = loss_sensor + loss_normals + loss_sil
        # total_loss = loss_sensor + loss_sil

        loss_dict = {
            'total_loss': total_loss,
            # 'loss_sil': loss_sil,
            # 'loss_depth': loss_depth,
            # 'loss_sensor': loss_sensor
        }

        return loss_dict

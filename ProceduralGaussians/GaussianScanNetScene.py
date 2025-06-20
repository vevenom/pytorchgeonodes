# Some parts of this code are based on shape-of-motion:
# https://github.com/vye16/shape-of-motion (Accessed Jun 18th 2025)
#
# The original license applies.

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from gsplat.rendering import rasterization
from pytorch3d.transforms import quaternion_apply
from pytorch3d.ops import knn_points
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.transforms.rotation_conversions import euler_angles_to_matrix
import lpips
from pytorch_msssim import SSIM
import functools
from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict

from DataRW.ScannotateDatasetMasks import ScannotateDatasetMasks
from DataRW.ScanNetRW import ScanNetRW
from SPSearch.utils import get_object_center
from SPSearch.utils import parse_params_from_json

from ProceduralGaussians.train_configs import FGLRConfig, BGLRConfig

from ProceduralGaussians.gaussian_scene_utils import forward_obj_gaussians
from ProceduralGaussians.GaussianNodes.utils import rgb_to_sh
from ProceduralGaussians.gaussian_metrics import ssim, psnr

def trimmed_l1_loss(pred, gt, quantile=0.9):
    loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss


def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        quantile_mask = (
            (sum_loss < torch.quantile(sum_loss, quantile)).squeeze(-1)
            if quantile < 1
            else torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)
        )
        ndim = sum_loss.shape[-1]
        if normalize:
            return torch.sum((sum_loss * mask)[quantile_mask]) / (
                ndim * torch.sum(mask[quantile_mask]) + 1e-8
            )
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])


class GaussianRenderer(nn.Module):
    def __init__(self, K, image_size, config):
        super().__init__()
        self.K = K
        self.image_size = image_size
        self.config = config

    def render(
            self,
            gaussians: dict,
            camera_pose,
            bg_color: torch.Tensor | float = 1.0,
            return_depth: bool = False,
    ) -> dict:

        C = camera_pose.shape[0]
        device = gaussians['means'].device

        N = gaussians['means'].shape[1]  # self.num_fg_gaussians if fg_only else self.num_gaussians
        sh_degree = self.config.gaussian_primitive.spherical_harmonics_degree

        means = gaussians['means'].view(-1, 3)  # (N, 3)
        quats = gaussians['quats'].view(-1, 4)  # (N, 4)

        if sh_degree is not None:
            colors = gaussians['colors'].view(-1, gaussians['colors'].shape[-2], 3)  # (N, K, 3)
        else:
            colors = gaussians['colors'].view(-1, 3)

        D = colors.shape[-1]

        scales = gaussians['scales'].view(-1, 3)
        opacities = gaussians['opacities'].view(-1)

        Ks = self.K[None].expand(C, -1, -1)

        if isinstance(bg_color, float):
            bg_color = torch.full((C, D), bg_color, device=device)
        assert isinstance(bg_color, torch.Tensor)

        mode = "RGB"
        output_split = [D]
        if return_depth:
            mode = "RGB+ED"
            output_split += [1]

        sh_degree = self.config.gaussian_primitive.spherical_harmonics_degree
        render_colors, alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            backgrounds=bg_color,
            viewmats=camera_pose,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            sh_degree=sh_degree,
            width=self.image_size[0],
            height=self.image_size[1],
            packed=False,
            render_mode=mode,
        )

        outputs = torch.split(render_colors, output_split, dim=-1)
        out_dict = {
            'img': outputs[0],
            "acc": alphas
        }
        if return_depth:
            out_dict['depth'] = outputs[1]

        return out_dict


class SceneModel(nn.Module):
    def __init__(
        self,
        scannotate_objects_instance,
        geometry_nodes,
        obj_idx,
        scannotate_config,
        scene_name,
        save_path,
        use_alternative_names = False,
        split_train_val = True,
    ):
        super().__init__()
        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.geometry_nodes = geometry_nodes
        self.scannotate_objects_instance = scannotate_objects_instance
        self.box_item = box_item = scannotate_objects_instance.obj_annotation_list[obj_idx]
        obj_center, obj_scale = get_object_center(box_item, return_scale=True) # We only need center
        self.obj_center = obj_center.to(device)

        self.scannet_instance = ScanNetRW(scannotate_config.scannet_processed_path, scannotate_config.scannet_ply_path,
                                          use_alternative_names=use_alternative_names)
        self.scannotate_masks_instance = scannotate_masks_instance = (
            ScannotateDatasetMasks(scannotate_config.scannotate_masks_path))
        scannotate_masks_instance.yield_obj_frames_dicts(self.scannet_instance, scene_name, obj_idx)
        self.scene_dict = scannotate_masks_instance.generate_obj_batch_dict(
            self.scannet_instance, box_item, scene_name, obj_idx, device)

        if split_train_val:
            print("Split training and validation views")
            self.scene_dict_train, self.scene_dict_eval = self.split_train_eval()

        self.save_path = save_path
        tensorboard_log_dir = os.path.join(save_path, "gaussian_training_logs/")
        self.writer = SummaryWriter(tensorboard_log_dir)

        # Configure learning rates
        self.fgd_lr_cfg = FGLRConfig()
        self.bgd_lr_cfg = BGLRConfig()

        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.ssim_depth = SSIM(data_range=1.0, size_average=True, channel=1)

    def split_train_eval(self):
        """
        Split images into training (0.66%) and validation (0.33%) set.
        """

        scene_dict_train = {}
        scene_dict_eval = {}
        eval_indices = torch.arange(0, self.scene_dict['color'].shape[0], step=3)
        train_indices = [i for i in range(self.scene_dict['color'].shape[0]) if i not in eval_indices]
        scene_dict_train['color'] = self.scene_dict['color'][train_indices]
        scene_dict_train['depth'] = self.scene_dict['depth'][train_indices]
        scene_dict_train['pose'] = self.scene_dict['pose'][train_indices]
        scene_dict_train['pose_renderer'] = self.scene_dict['pose_renderer'][train_indices]
        scene_dict_train['intrinsics'] = self.scene_dict['intrinsics'][train_indices]
        scene_dict_train['instance_seg'] = self.scene_dict['instance_seg'][train_indices]

        scene_dict_eval['color'] = self.scene_dict['color'][eval_indices]
        scene_dict_eval['depth'] = self.scene_dict['depth'][eval_indices]
        scene_dict_eval['pose'] = self.scene_dict['pose'][eval_indices]
        scene_dict_eval['pose_renderer'] = self.scene_dict['pose_renderer'][eval_indices]
        scene_dict_eval['intrinsics'] = self.scene_dict['intrinsics'][eval_indices]
        scene_dict_eval['instance_seg'] = self.scene_dict['instance_seg'][eval_indices]

        print("Train Indices: ", train_indices)
        print("Eval Indices: ", eval_indices)
        return scene_dict_train, scene_dict_eval

    def calculate_metrics(self, gaussians, lpips_fn):

        rgb_gt = self.scene_dict_eval['color']
        masked_rgb_gt = rgb_gt * self.scene_dict_eval['instance_seg'][..., None].float() / 255.0
        masked_rgb_gt = masked_rgb_gt.permute(0, 3, 1, 2)

        # Render the scene
        render_dict = self.render(gaussians, self.scene_dict_eval)
        masked_renders = render_dict['img'] * self.scene_dict_eval['instance_seg'][..., None].float()  # * 255.0
        masked_renders = masked_renders.permute(0, 3, 1, 2)

        ssims = []
        psnrs = []
        lpipss = []

        from tqdm import tqdm
        for idx in tqdm(range(masked_renders.shape[0]), desc="Metric evaluation progress"):
            ssims.append(
                ssim(masked_renders[idx][None, ...].contiguous(), masked_rgb_gt[idx][None, ...].contiguous()))
            psnrs.append(
                psnr(masked_renders[idx][None, ...].contiguous(), masked_rgb_gt[idx][None, ...].contiguous()))
            lpipss.append(lpips_fn(masked_renders[idx][None, ...].contiguous(),
                                   masked_rgb_gt[idx][None, ...].contiguous()).detach())

        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))

        results = {"SSIM": torch.tensor(ssims).mean().item(),
                   "PSNR": torch.tensor(psnrs).mean().item(),
                   "LPIPS": torch.tensor(lpipss).mean().item()}

        return results


    def calculate_colored_pcd(self, is_background=False, voxel_size=0.01):
        """
        Calculate colored point cloud from ScanNet views.
        """
        device = self.device

        color_img = self.scene_dict['color'] / 255.0

        poses = self.scene_dict['pose']

        depths = self.scene_dict['depth']
        image_edge_mask = self.scene_dict['image_edge_mask'].float() # (B, H, W, 1)
        depths = depths * image_edge_mask[..., 0]

        if is_background:
            masks = 1. - self.scene_dict['instance_seg']
        else:
            masks = self.scene_dict['instance_seg']

        intrinsics = self.scene_dict['intrinsics']

        image_size = color_img[0].shape[:2]
        image_size = (image_size[1], image_size[0])

        x_arange = torch.arange(0, image_size[0], device=device)
        y_arange = torch.arange(0, image_size[1], device=device)
        ys, xs = torch.meshgrid([y_arange, x_arange], indexing='ij')

        points_3d = torch.zeros((0, 3), device=device)
        colors = []
        for frame_ind in range(poses.shape[0]):
            pose = poses[frame_ind]

            depth_map = depths[frame_ind]
            color_map = color_img[frame_ind]
            mask = masks[frame_ind]

            mask = mask.view(-1)
            intrinsics_frame = intrinsics[frame_ind][:3, :3].to(torch.float32)
            intrinsics_inv = torch.linalg.inv(intrinsics_frame)

            zs = depth_map[ys, xs]
            cs = color_map[ys, xs, :]

            zs = zs.reshape((-1, 1)).transpose(1, 0)
            xs = xs.reshape((-1, 1)).transpose(1, 0)
            ys = ys.reshape((-1, 1)).transpose(1, 0)

            homo_ones = torch.ones_like(xs)
            xy_homo = torch.concatenate([xs, ys, homo_ones], dim=0).to(torch.float32)

            xyz_camera = zs * torch.matmul(intrinsics_inv, xy_homo)
            xyz_camera_homo = torch.concatenate([xyz_camera, homo_ones], dim=0)
            xyz_global = torch.matmul(pose, xyz_camera_homo)
            xyz_global = xyz_global.transpose(1, 0)[:, :-1]

            xyz_global = torch.round(xyz_global / voxel_size) * voxel_size

            xyz_global = xyz_global[torch.logical_and(mask == 1, zs[0, :] > 0), :]

            cs = cs.view(-1, 3)
            cs = cs[torch.logical_and(mask == 1, zs[0, :] > 0), :]

            xyz_global_unique, xyz_global_object_counts = torch.unique(xyz_global, return_counts=True, dim=0)

            original_knn = knn_points(xyz_global_unique[None], xyz_global[None])
            original_knn_ind = original_knn.idx[0,...,0]

            cs = cs[original_knn_ind]
            colors.append(cs)

            points_3d = torch.concatenate([points_3d, xyz_global_unique],
                                          dim=0)


        colors = torch.cat(colors, dim=0)
        xyz_colors = torch.cat([points_3d, colors], dim=-1)

        return xyz_colors

    def render(
            self,
            gaussians: dict,
            scene_dict,
            bg_color: torch.Tensor | float = 0.0,
            return_depth: bool = False,
    ) -> dict:
        """
        Render scene with given Gaussians.
        """

        w2cs = torch.linalg.inv(scene_dict['pose'])
        sh_degree = self.geometry_nodes.config.gaussian_primitive.spherical_harmonics_degree

        device = gaussians['means'].device
        C = w2cs.shape[0]

        gt_imgs = scene_dict['color'].cpu().numpy()
        image_size = (gt_imgs[0].shape[1], gt_imgs[0].shape[0])
        W, H = image_size

        N = gaussians['means'].shape[1]  # self.num_fg_gaussians if fg_only else self.num_gaussians

        means = gaussians['means'].view(-1, 3)  # (N, 3)
        quats = gaussians['quats'].view(-1, 4)  # (N, 4)

        if sh_degree is not None:
            colors = gaussians['colors'].view(-1, gaussians['colors'].shape[-2], 3)  # (N, K, 3)
        else:
            colors = gaussians['colors'].view(-1, 3)  # (N, 3)

        D = colors.shape[-1]

        scales = gaussians['scales'].view(-1, 3)
        opacities = gaussians['opacities'].view(-1)

        Ks = scene_dict['intrinsics'][:, :3, :3].float()  # (C, 3, 3)

        if isinstance(bg_color, float):
            bg_color = torch.full((C, D), bg_color, device=device)
        assert isinstance(bg_color, torch.Tensor)

        mode = "RGB"
        output_split = [D]
        if return_depth:
            mode = "RGB+ED"
            output_split += [1]

        render_colors, alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            backgrounds=bg_color,
            viewmats=w2cs,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            sh_degree=sh_degree,
            width=W,
            height=H,
            packed=False,
            render_mode=mode,
        )

        outputs = torch.split(render_colors, output_split, dim=-1)
        out_dict = {
            'img': outputs[0],
            "acc": alphas
        }
        if return_depth:
            out_dict['depth'] = outputs[1]

        return out_dict

    def prepare_loss(self):
        gt_depths = self.scene_dict['depth']
        depth_hole_mask = (gt_depths > 0).float()

        self.scene_dict['depth_loss_mask'] = depth_hole_mask[..., None]

        edge_size = 5 # Mask edges of ScanNet images
        image_edge_mask = torch.ones_like(gt_depths)
        image_edge_mask[:, :edge_size] = 0
        image_edge_mask[:, :, :edge_size] = 0
        image_edge_mask[:, image_edge_mask.shape[1] - edge_size:,] = 0
        image_edge_mask[:, :, image_edge_mask.shape[2] - edge_size:] = 0

        self.scene_dict['image_edge_mask'] = image_edge_mask[..., None]

    def init_shape_from_params(self, input_params_dict, rotation_matrix, translation_offset=None):
        """
        Initialize Gaussians using geometry nodes and shape parameters.
        """
        return forward_obj_gaussians(self.geometry_nodes, input_params_dict, rotation_matrix, translation_offset,
                                     self.obj_center)

    def compute_loss_3d(self, obj_gaussians, gaussians, target_colored_pcd,
                        cd_loss_w,
                        gs_sampling_scale,
                        n_samples=10000
                        ):
        """
        Chamfer distance between object mesh and object Gaussians.
        """

        obj_mesh_init = obj_gaussians.get_mesh(use_offsets=False)

        mesh_pcd_init = obj_mesh_init.verts_packed()[None]

        means = gaussians['means']
        B, M, _ = means.shape
        random_indices = torch.randint(low=0, high=M, size=(n_samples,))

        sampled_means = means[:, random_indices]
        sampled_quads = gaussians['quats'][:, random_indices]
        sampled_scales = gaussians['scales'][:, random_indices]

        sampled_points = sampled_means + quaternion_apply(
            sampled_quads,
            gs_sampling_scale * sampled_scales * torch.randn_like(sampled_means))

        (cd_loss_x, cd_loss_y) = chamfer_distance(sampled_points, mesh_pcd_init,
                                                  batch_reduction=None,
                                                  point_reduction=None,
                                                  # x_normals=self.surface_points_normals[None],
                                                  # y_normals=mesh_normals,
                                                  norm=2,
                                                  single_directional=False)[0]  # / self.cd_clamp

        cd_loss_init = 1.0 * cd_loss_x.mean() + 1.0 * cd_loss_y.mean()

        loss = cd_loss_w * cd_loss_init

        loss_3d_dict = {
            'cd_loss_init': cd_loss_init,
        }

        return loss, loss_3d_dict

    def compute_loss_with_bgd(self, gaussians, bgd_gaussians, render_seperately=False):
        """
        Calculate render-and-compare loss with BGD Gaussians.
        """
        if not render_seperately:
            final_gaussians = {}
            for key in gaussians.keys():
                final_gaussians[key] = torch.cat([gaussians[key], bgd_gaussians[key]], dim=1)

            render_dict = self.render(final_gaussians, self.scene_dict, return_depth=True)
        else:
            render_dict_bgd = self.render(bgd_gaussians, self.scene_dict, return_depth=True)
            render_dict_fgd = self.render(gaussians, self.scene_dict, return_depth=True)

            render_dict = {}

            render_dict['img'] = render_dict_bgd['img'] + render_dict_fgd['img']
            render_dict['depth'] = render_dict_bgd['depth'] + render_dict_fgd['depth']


        # RGB Loss
        rendered_imgs = render_dict['img']  # (B, H, W, 3)
        gt_imgs = (self.scene_dict['color'] / 255.).float()  # (B, H, W, 3)

        depth_loss_masks = self.scene_dict['depth_loss_mask'].float() # (B, H, W, 1)
        image_edge_mask = self.scene_dict['image_edge_mask'].float() # (B, H, W, 1)


        gt_imgs = gt_imgs * image_edge_mask
        rendered_imgs = rendered_imgs * image_edge_mask

        render_dict['rendered_imgs'] = rendered_imgs
        render_dict['gt_imgs_masked'] = gt_imgs

        rgb_loss = 0.8 * F.l1_loss(rendered_imgs, gt_imgs) + 0.2 * (
                    1 - self.ssim(rendered_imgs.permute(0, 3, 1, 2), gt_imgs.permute(0, 3, 1, 2)))

        # Depth Loss
        gt_depths = self.scene_dict['depth'][..., None] * depth_loss_masks #* color_loss_masks # (B, H, W)

        render_dict['depth'] = render_dict['depth'] + torch.zeros_like(
            render_dict['depth'])  # a view is being modified inplace.
        depths = render_dict['depth'] * depth_loss_masks  # * seg_masks[..., None] # (B, H, W)
        gt_depths = gt_depths * depth_loss_masks
        render_dict['gt_depths_masked'] = gt_depths

        depth_loss = masked_l1_loss(depths, gt_depths, mask=None, quantile=0.98)

        return rgb_loss, depth_loss, render_dict

    def compute_loss_wo_bgd(self, gaussians):
        """
        Calculate render-and-compare loss without BGD Gaussians.
        """

        # Render Images
        render_dict = self.render(gaussians, self.scene_dict, return_depth=True)

        # RGB Loss
        rendered_imgs = render_dict['img']  # (B, H, W, 3)
        gt_imgs = (self.scene_dict['color'] / 255.).float()  # (B, H, W, 3)

        # Segment out the foreground object
        seg_masks = self.scene_dict['instance_seg'].float()[..., None]  # (B, H, W, 1)
        image_edge_mask = self.scene_dict['image_edge_mask'].float() # (B, H, W, 1)
        seg_masks = seg_masks * image_edge_mask

        depth_loss_masks = self.scene_dict['depth_loss_mask'].float() * seg_masks  # (B, H, W, 1)

        gt_imgs = gt_imgs * seg_masks
        rendered_imgs = rendered_imgs * seg_masks

        render_dict['rendered_imgs'] = rendered_imgs
        render_dict['gt_imgs_masked'] = gt_imgs

        rgb_loss = 0.8 * F.l1_loss(rendered_imgs, gt_imgs) + 0.2 * (
                    1 - self.ssim(rendered_imgs.permute(0, 3, 1, 2), gt_imgs.permute(0, 3, 1, 2)))

        # Depth Loss
        gt_depths = self.scene_dict['depth'][..., None]  # * depth_loss_masks #* color_loss_masks # (B, H, W)

        render_dict['depth'] = render_dict['depth'] + torch.zeros_like(
            render_dict['depth'])  # a view is being modified inplace.

        depths = render_dict['depth'] * depth_loss_masks
        gt_depths = gt_depths * depth_loss_masks
        render_dict['gt_depths_masked'] = gt_depths

        # pred_disp = 1.0 / (depths + 1e-5)
        # gt_disp = 1.0 / (gt_depths + 1e-5)
        depth_loss = masked_l1_loss(depths, gt_depths, mask=None, quantile=0.98)

        return rgb_loss, depth_loss, render_dict

    def compute_regularization_losses_bgd(self, final_bgd_gaussians):
        """
        Calculate regularization losses for BGD Gaussians.
        """
        # Mean Regularization Loss
        bgd_means = final_bgd_gaussians['means'].view(-1, 3)
        bgd_means_init = self.initial_bgd_gaussians['means'].view(-1, 3)
        means_reg_loss = (bgd_means_init - bgd_means).norm(p=2, dim=-1).mean()

        # Scale Regularization Loss
        bgd_scales = final_bgd_gaussians['scales'].view(-1, 3)
        scales_reg_loss = bgd_scales.norm(p=2, dim=-1).mean()

        colors_reg_loss = (final_bgd_gaussians['colors'] - self.initial_bgd_gaussians['colors']).norm(p=2, dim=-1).mean()

        return means_reg_loss, scales_reg_loss, colors_reg_loss

    def compute_loss(self, obj_gaussians, target_colored_pcd, use_bgd=False):
        loss = 0.0

        loss_config = self.geometry_nodes.config.losses
        
        init_gaussians, gaussians = obj_gaussians.compute_gaussians()
        params = obj_gaussians.params

        rgb_loss_fgd = torch.tensor(0)
        rgb_loss_bgd = torch.tensor(0)
        depth_loss_fgd = torch.tensor(0)
        depth_loss_bgd = torch.tensor(0)
        bgd_means_reg_loss = torch.tensor(0)
        bgd_scales_reg_loss = torch.tensor(0)
        bgd_colors_reg_loss = torch.tensor(0)
        if use_bgd:
            final_bgd_gaussians = self.get_final_bgd_gaussians()
            rgb_loss_bgd, depth_loss_bgd, render_dict_bgd = self.compute_loss_with_bgd(gaussians, final_bgd_gaussians)
            (bgd_means_reg_loss,
             bgd_scales_reg_loss,
             bgd_colors_reg_loss) = self.compute_regularization_losses_bgd(final_bgd_gaussians)

            bgd_loss = 0.
            bgd_loss += bgd_means_reg_loss * loss_config.means_offsets_loss_w
            bgd_loss += bgd_scales_reg_loss * loss_config.scales_offsets_loss_w
            bgd_loss += bgd_colors_reg_loss * loss_config.colors_offsets_loss_w

            bgd_loss += rgb_loss_bgd * loss_config.rgb_loss_w
            bgd_loss += depth_loss_bgd * loss_config.depth_loss_w

            loss += bgd_loss * loss_config.bgd_loss_w

            rgb_loss_fgd, depth_loss_fgd, render_dict = self.compute_loss_wo_bgd(gaussians)
            render_dict['img_bgd'] = render_dict_bgd['img']
            loss += rgb_loss_fgd * loss_config.rgb_loss_w
            loss += depth_loss_fgd * loss_config.depth_loss_w

            rgb_loss = rgb_loss_fgd + rgb_loss_bgd
            depth_loss = depth_loss_fgd + depth_loss_bgd
        else:
            rgb_loss, depth_loss, render_dict = self.compute_loss_wo_bgd(gaussians)

            loss += rgb_loss * loss_config.rgb_loss_w
            loss += depth_loss * loss_config.depth_loss_w

        # Mean Regularization Loss
        means_offsets = obj_gaussians.get_means_offsets(use_transform=False).view(-1, 3)

        means_offsets_reg_loss = means_offsets.norm(p=2, dim=-1).mean()
        loss += means_offsets_reg_loss * loss_config.means_offsets_loss_w

        verts_offsets = obj_gaussians.get_verts_offsets(use_transform=False).view(-1, 3)
        verts_offsets_reg_loss = verts_offsets.norm(p=2, dim=-1).mean()
        loss += verts_offsets_reg_loss * loss_config.verts_offsets_loss_w

        # Scale Regularization Loss
        scales_offsets = torch.cat([s_offs for s_offs in params["scales_offsets"]], dim=1).view(-1, 2)
        scales_offsets_reg_loss = scales_offsets.norm(p=1, dim=-1).mean()
        loss += scales_offsets_reg_loss * loss_config.scales_offsets_loss_w

        color_offsets_reg_loss = 0.
        for c_offs in params["colors_offsets"]:
            c_offs = c_offs.view(-1, 3)
            color_offsets_reg_loss += torch.var(c_offs, dim=0).mean()
        color_offsets_reg_loss /= len(params["colors_offsets"])
        loss += color_offsets_reg_loss * loss_config.colors_offsets_loss_w

        opacities = gaussians['opacities']
        opacities_loss = (opacities).sum() / init_gaussians['means'].shape[1]
        loss += opacities_loss * loss_config.opacity_loss_w

        opacities_mean = opacities.mean()
        opacities_min = opacities.min()
        opacities_max = opacities.max()

        loss_3d, loss_3d_dict = self.compute_loss_3d(obj_gaussians, gaussians, target_colored_pcd,
                                       cd_loss_w=loss_config.loss_3d_cd_w,
                                       gs_sampling_scale=loss_config.loss_3d_gs_sampling_scale)
        loss += loss_3d

        loss_dict = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'rgb_loss_fr': rgb_loss_fgd,
            'rgb_loss_bgd': rgb_loss_bgd,
            'depth_loss': depth_loss,
            'depth_loss_fr': depth_loss_fgd,
            'depth_loss_bgd': depth_loss_bgd,
            'means_offsets_loss': means_offsets_reg_loss,
            'verts_offsets_reg_loss': verts_offsets_reg_loss,
            'scales_offsets_loss': scales_offsets_reg_loss,
            'color_offsets_reg_loss': color_offsets_reg_loss,
            'opacities_loss': opacities_loss,
            'opacities_mean': opacities_mean,
            'opacities_min': opacities_min,
            'opacities_max': opacities_max,
            'loss_3d': loss_3d,
        }
        loss_dict.update(loss_3d_dict)

        if use_bgd:
            loss_dict['bgd_means_reg_loss'] = bgd_means_reg_loss
            loss_dict['bgd_scales_reg_loss'] = bgd_scales_reg_loss
            loss_dict['bgd_colors_reg_loss'] = bgd_colors_reg_loss

        return loss, loss_dict, render_dict

    def initialize_from_colored_pcd(self, xyz_colors):
        """
        Initialize Gaussians from colored pcd.
        """
        sampled_xyz = xyz_colors[:, :3][None]
        sampled_colors = xyz_colors[:, 3:][None]
        sampled_colors = torch.logit(sampled_colors)

        means = sampled_xyz
        colors = sampled_colors

        config = self.geometry_nodes.config

        if (config.gaussian_primitive.spherical_harmonics_degree is not None and
                config.gaussian_primitive.spherical_harmonics_degree > 0):
            final_colors = torch.zeros((
                1, colors.shape[1], (config.gaussian_primitive.spherical_harmonics_degree + 1) ** 2, 3),
                device=colors.device)  # [1, N, K, 3]
            final_colors[:, :, 1] = rgb_to_sh(colors)
            colors = final_colors

        B, M, _ = means.shape

        scales = torch.ones_like(means) * self.geometry_nodes.config.bgd_gaussians.bgd_gaussians_init_scale

        quats = torch.zeros((B, M, 4), device=means.device)
        quats[..., 0] = 1

        opacities = torch.ones_like(means[..., :1]) * self.geometry_nodes.config.bgd_gaussians.init_opacity

        optimize_bgd = self.geometry_nodes.config.bgd_gaussians.optimize_bgd

        means = nn.Parameter(means, requires_grad=optimize_bgd)
        colors = nn.Parameter(colors, requires_grad=optimize_bgd)
        scales = nn.Parameter(scales, requires_grad=optimize_bgd)
        quats = nn.Parameter(quats, requires_grad=optimize_bgd)
        opacities = nn.Parameter(opacities, requires_grad=False)

        gaussians = {
            'means': means,
            'scales': scales,
            'quats': quats,
            'colors': colors,
            'opacities': opacities
        }

        return gaussians

    def get_final_bgd_gaussians(self):
        """
        Get final BGD Gaussians.
        """
        final_bgd_gaussians = {}
        final_bgd_gaussians['means'] = self.bgd_gaussians['means']
        final_bgd_gaussians['scales'] = self.bgd_gaussians['scales']
        final_bgd_gaussians['quats'] = self.bgd_gaussians['quats']
        final_bgd_gaussians['opacities'] = self.bgd_gaussians['opacities']
        final_bgd_gaussians['colors'] = torch.sigmoid(self.bgd_gaussians['colors'])

        return final_bgd_gaussians

    def optimize(self, obj_json):
        """
        Optimize procedural Gaussian Splatting using render-and-compare.
        """

        # Get object parameters
        input_params_dict, rot_matrix, translation_offset = parse_params_from_json(obj_json, self.device)
        self.prepare_loss()

        # Initialize object Gaussians using procedural model and input parameters
        obj_gaussians, obj_mesh = (
            self.init_shape_from_params(input_params_dict, rot_matrix, translation_offset[None]))

        use_bgd=self.geometry_nodes.config.bgd_gaussians.use_bgd_gaussians

        # Initialize Gaussian colors from colored scene pcd
        with torch.no_grad():
           obj_xyz_colors = self.calculate_colored_pcd(voxel_size=0.01)
           init_gaussians = obj_gaussians.initialize_from_colored_pcd(obj_xyz_colors)
           obj_gaussians.assign_colors(init_gaussians['colors'])

           if use_bgd:
               bgd_xyz_colors = self.calculate_colored_pcd(is_background=True, voxel_size=0.02)
               bf_knn = knn_points(bgd_xyz_colors[None, :, :3], obj_xyz_colors[None, :, :3])

               bf_dists = bf_knn.dists[0]
               bgd_xyz_colors = bgd_xyz_colors[bf_dists[..., 0] > 0.001, :]

               self.bgd_gaussians = self.initialize_from_colored_pcd(bgd_xyz_colors)
               self.initial_bgd_gaussians = self.get_final_bgd_gaussians()

        # Calculate initial metrics
        with torch.no_grad():
            lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
            gaussians = obj_gaussians.compute_final_gaussians(init_gaussians)
            results_before = self.calculate_metrics(gaussians, lpips_fn)

        # Set optim and schedulers
        optimizers, schedulers = self.configure_optimizers(obj_gaussians, use_bgd=use_bgd)

        print('Results before')
        print(results_before)

        num_iters = 300 # NOTE: High number causes overfit!
        log_freq = 30
        for global_step in range(num_iters):

            # Compute the loss
            loss, loss_dict, render_dict = self.compute_loss(obj_gaussians, obj_xyz_colors,
                                                             use_bgd=use_bgd)
            loss.backward()

            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            for sched in schedulers.values():
                sched.step()

            if global_step % log_freq == 0:
                self.log_and_validate(loss_dict, render_dict,
                         optimizers, schedulers, global_step,
                         input_params_dict, rot_matrix, translation_offset, results_before)

        self.save_checkpoint(optimizers, schedulers, global_step)

        self.writer.flush()
        self.writer.close()

    def log_and_validate(self, loss_dict, render_dict,
                         optimizers, schedulers, global_step,
                         input_params_dict, rot_matrix, translation_offset, results_before):
        print("Iter: ", global_step)
        for k, v in loss_dict.items():
            print(k, ':', v.item())

            self.writer.add_scalar(k, v.item(), global_step)

        self.writer.add_images('color', render_dict['img'][::8], global_step, dataformats="NHWC")
        if self.geometry_nodes.config.bgd_gaussians.use_bgd_gaussians:
            self.writer.add_images('color_bgd', render_dict['img_bgd'][::8],
                                   global_step, dataformats="NHWC")

        self.writer.add_images('rendered_imgs',
                               render_dict['rendered_imgs'][::8], global_step, dataformats="NHWC")
        self.writer.add_images('depth', render_dict['depth'][::8] / render_dict['depth'][:8].max(),
                               global_step, dataformats="NHWC")

        self.writer.add_images(
            'depth_gt',
            (render_dict['gt_depths_masked'][::8] /
             render_dict['gt_depths_masked'][:8].max()),
            global_step, dataformats="NHWC")

        self.writer.add_images(
            'color_gt',
            render_dict['gt_imgs_masked'][::8],
            global_step, dataformats="NHWC")

        obj_gaussians, obj_mesh = (
            forward_obj_gaussians(self.geometry_nodes, input_params_dict, rot_matrix, translation_offset[None],
                                  obj_center=self.obj_center, transform2blender_coords=True))

        init_gaussians, final_gaussians = obj_gaussians.compute_gaussians()

        init_camera_pose = torch.eye(4, device=self.device)[None]
        init_camera_pose[:, :3, 3] += self.obj_center[0]
        init_camera_pose = torch.linalg.inv(init_camera_pose)

        nvs_rot_matrix = torch.eye(4, device=self.device)[None]
        nvs_rot_matrix[:, :3, :3] = euler_angles_to_matrix(torch.tensor([3.5, 0, 0]),
                                                           convention='XYZ')

        curr_camera_pose = torch.bmm(nvs_rot_matrix, init_camera_pose)
        curr_camera_pose[:, 2, 3] += 3

        custom_render_image = self.log_render(final_gaussians, curr_camera_pose)

        self.writer.add_images(
            'color_nvs',
            custom_render_image,
            global_step, dataformats="NHWC")

        self.save_checkpoint(optimizers, schedulers, global_step)

        with torch.no_grad():
            lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
            gaussians = obj_gaussians.compute_final_gaussians(init_gaussians)
            results_curr = self.calculate_metrics(gaussians, lpips_fn)

        print('Results before')
        print(results_before)
        print('Results curr')
        print(results_curr)

    def log_render(self, final_gaussians, pose):

        K = torch.tensor([
            [577.5907, 0.0000, 318.9054],
            [0.0000, 578.7298, 242.6836],
            [0.0000, 0.0000, 1.0000]],
            device=pose.device
        )

        renderer = GaussianRenderer(K=K, image_size=(640, 480), config=self.geometry_nodes.config)

        rendered_gaussian_dict = renderer.render(final_gaussians, pose)

        rendered_color = rendered_gaussian_dict['img']

        return rendered_color

    def save_checkpoint(self, optimizers, schedulers, global_step):

        gaussians_path = os.path.join(self.save_path, 'gaussian_params.pth')
        self.geometry_nodes.save_trainable_params(gaussians_path)

        obj_center_path = os.path.join(self.save_path, 'obj_center.pth')
        torch.save(self.obj_center, obj_center_path)


        optimizer_dict = {k: v.state_dict() for k, v in optimizers.items()}
        scheduler_dict = {k: v.state_dict() for k, v in schedulers.items()}
        ckpt = {
            "optimizers": optimizer_dict,
            "schedulers": scheduler_dict,
            "global_step": global_step,
        }

        training_meta_path = os.path.join(self.save_path, "training_meta.pth")
        torch.save(ckpt, training_meta_path)

    def configure_optimizers(self, obj_gaussians, use_bgd=False):
        fgd_lr_dict = asdict(self.fgd_lr_cfg)
        optimizers = {}
        schedulers = {}

        for name, params in obj_gaussians.params.items():
            lr = fgd_lr_dict[name]

            if isinstance(params, nn.ModuleList):
                params = params.parameters()
                optim = torch.optim.Adam(params=params, lr=lr)
            else:
                optim = torch.optim.Adam([{"params": params, "lr": lr, "name": name}])
            fnc = lambda _, **__: 1.0

            optimizers[name] = optim
            schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                optim, functools.partial(fnc, lr_init=lr)
            )

        bgd_lr_dict = asdict(self.bgd_lr_cfg)
        if use_bgd:
            for name, params in self.bgd_gaussians.items():
                lr = bgd_lr_dict[name]
                optim = torch.optim.Adam([{"params": params, "lr": lr, "name": name}])
                fnc = lambda _, **__: 1.0

                optimizers[name] = optim
                schedulers[name] = torch.optim.lr_scheduler.LambdaLR(
                    optim, functools.partial(fnc, lr_init=lr)
                )

        return optimizers, schedulers
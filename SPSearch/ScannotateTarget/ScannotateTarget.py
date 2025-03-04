import torch
import numpy as np
import cv2
import copy
import os

from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import Translate
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.ops import knn_points

from DataRW.ScanNetRW import ScanNetRW
from DataRW.ScannotateDatasetMasks import ScannotateDatasetMasks

from SPSearch.Target import Target
from SPSearch.OcclusionGrid import OcclusionGrid
from PytorchGeoNodes.Pytorch3DRenderer.Torch3DRenderer import Torch3DRenderer
from PytorchGeoNodes.utils import colormap
from SPSearch.utils import get_object_center


class ScannotateTarget(Target):
    def __init__(self, scannotate_objects_instance,
                 scannotate_config, use_alternative_names,
                 geometry_nodes, scene_name, obj_idx,
                 use_object_scale=False,
                 optimize_translation=True,
                 log_path='outputs/demo'):

        super().__init__(log_path)

        self.log_path = log_path

        self.geometry_nodes = geometry_nodes

        self.scannotate_objects_instance = scannotate_objects_instance
        self.shapenet_path = ''

        self.scannet_instance = ScanNetRW(scannotate_config.scannet_processed_path, scannotate_config.scannet_ply_path,
                                          use_alternative_names=use_alternative_names)

        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.box_item = box_item = scannotate_objects_instance.obj_annotation_list[obj_idx]

        obj_center, obj_scale = get_object_center(box_item, return_scale=True) # We only need center
        obj_center = obj_center.to(self.device)

        self.obj_center = obj_center.to(device)
        self.use_object_scale = use_object_scale
        self.optimize_translation = optimize_translation

        self.scannotate_masks_instance = scannotate_masks_instance = (
            ScannotateDatasetMasks(scannotate_config.scannotate_masks_path))
        scannotate_masks_instance.yield_obj_frames_dicts(self.scannet_instance, scene_name, obj_idx)

        def gen_scene_frame_iterator():
            return scannotate_masks_instance.yield_obj_frames_dicts(self.scannet_instance, scene_name, obj_idx)

        self.scene_frame_iterator_fn = gen_scene_frame_iterator

        self.scene_dict = scannotate_masks_instance.generate_obj_batch_dict(
            self.scannet_instance, box_item, scene_name, obj_idx, device)

        self.scene_pcd = self.scene_dict['object_points'][None]
        self.other_objs_pcd = self.scene_dict['other_object_points'][None]

        self.occlusion_grid = OcclusionGrid(surface_points=self.scene_pcd[0],
                                            other_surface_points=self.other_objs_pcd[0])
        self.occlusion_grid.to(device)

        self.occlusion_grid.calculate_grid(self.scene_dict)

        print('ScannotateTarget initialized with scene: {}, obj_idx: {}'.format(scene_name, obj_idx))

    def get_scene_pcd(self):
        return self.scene_pcd

    def get_obj_pose(self):
        transform = self.box_item.transform3d.to(self.device)

        return transform.get_matrix()

    def calculate_colored_pcd(self, use_mask=True, voxel_size=0.01):

        device = self.device

        color_img = self.scene_dict['color'] / 255.0

        poses = self.scene_dict['pose']

        depths = self.scene_dict['depth']

        if not use_mask:
            masks = torch.ones_like(self.scene_dict['instance_seg'])
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

    def calculate_mesh_from_input_dict(self, input_params_dict, rotation_matrix, translation_offset=None,
                                       return_geo_node_mesh=False):
        device = self.device

        _, outputs = self.geometry_nodes.forward(input_params_dict, transform2blender_coords=True)
        geo_node_obj_mesh = outputs[0][0][0]
        obj_mesh = Meshes(verts=geo_node_obj_mesh.verts, faces=geo_node_obj_mesh.faces)

        verts = obj_mesh.verts_packed()
        faces = obj_mesh.faces_packed()

        transform = Transform3d(device=device)

        if rotation_matrix is not None:
            transform_rot = Transform3d(matrix=rotation_matrix, device=device)
            assert rotation_matrix.shape[0] == 1
            transform = transform.compose(transform_rot)
        else:
            assert False, "We are doing experiments with rotations"

        # If object is not at origin
        bb = obj_mesh.get_bounding_boxes()  # (N, 3, 2)

        bb_center = (bb[:, :, 1] - bb[:, :, 0]) / 2 + bb[:, :, 0]

        verts = verts - bb_center

        verts = transform.transform_points(verts)

        translation = self.obj_center #- bb_center
        verts = verts + translation

        if translation_offset is not None:
            translate = Translate(translation_offset)
            verts = translate.transform_points(verts)

        obj_mesh = Meshes(verts=[verts], faces=[faces], textures=obj_mesh.textures)

        if return_geo_node_mesh:
            return obj_mesh, geo_node_obj_mesh
        return obj_mesh

    def calculate_cost_from_input_dict(self, input_params_dict, rotation_matrix, translation_offset=None,
                                       use_fast_loss=False, update_negative_points=False):

        obj_mesh = self.calculate_mesh_from_input_dict(input_params_dict, rotation_matrix, translation_offset)
        loss = self.occlusion_grid.calculate_loss_from_mesh(obj_mesh)

        return loss

    def log_iter_from_input_dict(self, input_params_dict, rotation_matrix,
                                 translation_offset=None, iter_num=0, file_prefix='',
                                 transparent_overlay=True, color_mode=None):
        """
        color_mode can be None or 'base', 'individual'

        """
        device = self.device
        logger_image_downscale = 2

        obj_mesh, geo_node_obj_mesh = (
            self.calculate_mesh_from_input_dict(input_params_dict, rotation_matrix, translation_offset,
                                                return_geo_node_mesh=True))

        frame_log_freq = 1
        for scene_frame_ind in range(0, self.scene_dict['color'].shape[0], frame_log_freq):
            color_img = self.scene_dict['color'][scene_frame_ind] / 255.0
            color_img = color_img.cpu().numpy()

            image_size = (color_img.shape[1] // logger_image_downscale, color_img.shape[0] // logger_image_downscale)
            color_img = cv2.resize(color_img, image_size)
            color_img[..., [2,1,0]] = color_img[..., [0,1,2]]
            pose = self.scene_dict['pose_renderer'][scene_frame_ind][None]
            intrinsics = self.scene_dict['intrinsics'][scene_frame_ind][None]
            intrinsics = intrinsics.clone()
            intrinsics[:, 0, 0] /= logger_image_downscale
            intrinsics[:, 1, 1] /= logger_image_downscale
            intrinsics[:, 0, 2] /= logger_image_downscale
            intrinsics[:, 1, 2] /= logger_image_downscale

            image_size = color_img.shape[:2]
            renderer_textureless = Torch3DRenderer().create_renderer(
                pose,
                intrinsics,
                image_size,
                z_clip_value=0.1,
                texturless=True,
                device=device)
            rendered_mesh_sil, zbuf = renderer_textureless(obj_mesh)
            rendered_mesh_mask_vis = rendered_mesh_sil[0, ..., :3].detach().cpu().numpy()

            renderer_texture = Torch3DRenderer().create_renderer(
                pose,
                intrinsics,
                image_size,
                z_clip_value=0.1,
                texturless=False,
                device=device)

            verts = obj_mesh.verts_packed()
            faces = obj_mesh.faces_packed()
            # create a simple gray texture for mesh
            if color_mode is None:
                verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
            elif color_mode == 'base':
                base_prim_ids_np = geo_node_obj_mesh.verts_base_primitive_ids.detach().cpu().numpy().astype(np.int32)[0]
                colors_np = colormap[base_prim_ids_np + 1]
                verts_rgb = torch.tensor(colors_np, dtype=verts.dtype, device=verts.device)[None]  # (1, V, 3)
            elif color_mode == 'individual':
                ind_prim_ids_np = geo_node_obj_mesh.verts_individual_primitive_ids.detach().cpu().numpy().astype(np.int32)[0]
                colors_np = colormap[ind_prim_ids_np + 1]
                verts_rgb = torch.tensor(colors_np, dtype=verts.dtype, device=verts.device)[None]  # (1, V, 3)
            else:
                assert False, 'Color mode {} not supported'.format(color_mode)

            textures = TexturesVertex(verts_features=verts_rgb.to(device))

            textured_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
            rendered_textured_mesh, _ = renderer_texture(textured_mesh)
            rendered_mesh_texture_vis = rendered_textured_mesh[0, ..., :3].cpu().numpy()

            rendered_color_overlaid = copy.deepcopy(color_img)

            if not transparent_overlay:
                rendered_color_overlaid[rendered_mesh_mask_vis[:, :, 2] == 1, :] = \
                    (
                        rendered_mesh_texture_vis)[rendered_mesh_mask_vis[:, :, 2] == 1, :]
            else:
                rendered_color_overlaid[rendered_mesh_mask_vis[:, :, 2] == 1, :] = \
                    color_img[rendered_mesh_mask_vis[:, :, 2] == 1, :] * 0.3 + (
                        (rendered_mesh_texture_vis[rendered_mesh_mask_vis[:, :, 2] == 1, :]) * 0.7)


            vis_img = np.concatenate((color_img, rendered_color_overlaid), axis=1)

            cv2.imwrite(os.path.join(self.log_path, file_prefix + 'vis_{:05d}_'.format(iter_num) +
                        self.scene_dict['frame_name'][scene_frame_ind] + '.jpg'), vis_img * 255)

    def render_image(self, obj_mesh,
                     scene_frame_ind=0, transparent_overlay=False):

        device = self.device
        logger_image_downscale = 2

        color_img = self.scene_dict['color'][scene_frame_ind] / 255.0
        color_img = color_img.cpu().numpy()

        image_size = (color_img.shape[1] // logger_image_downscale, color_img.shape[0] // logger_image_downscale)
        color_img = cv2.resize(color_img, image_size)
        # depth_img = self.scene_dict['depth'][scene_frame_ind]
        pose = self.scene_dict['pose_renderer'][scene_frame_ind][None]
        intrinsics = self.scene_dict['intrinsics'][scene_frame_ind][None]
        intrinsics = intrinsics.clone()
        intrinsics[:, 0, 0] /= logger_image_downscale
        intrinsics[:, 1, 1] /= logger_image_downscale
        intrinsics[:, 0, 2] /= logger_image_downscale
        intrinsics[:, 1, 2] /= logger_image_downscale

        image_size = color_img.shape[:2]
        renderer_textureless = Torch3DRenderer().create_renderer(
            pose,
            intrinsics,
            image_size,
            z_clip_value=0.1,
            texturless=True,
            device=device)
        rendered_mesh_sil, zbuf = renderer_textureless(obj_mesh)
        rendered_mesh_mask_vis = rendered_mesh_sil[0, ..., :3].detach().cpu().numpy()

        renderer_texture = Torch3DRenderer().create_renderer(
            pose,
            intrinsics,
            image_size,
            z_clip_value=0.1,
            texturless=False,
            device=device)

        verts = obj_mesh.verts_packed()
        faces = obj_mesh.faces_packed()
        # create a simple gray texture for mesh
        verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(device))

        vis_z_buf = False
        if not vis_z_buf:
            textured_mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
            rendered_textured_mesh, _ = renderer_texture(textured_mesh)
            rendered_mesh_texture_vis = rendered_textured_mesh[0, ..., :3].cpu().numpy()
        else:
            rendered_mesh_texture_vis = zbuf.cpu().numpy()[0, ..., 0]
            z_min_val = rendered_mesh_texture_vis[rendered_mesh_mask_vis[:, :, 2] == 1].min()
            rendered_mesh_texture_vis = np.clip(rendered_mesh_texture_vis, a_min=z_min_val, a_max=None)

            rendered_mesh_texture_vis = (rendered_mesh_texture_vis - z_min_val) / \
                                        (rendered_mesh_texture_vis.max() - z_min_val)
            rendered_mesh_texture_vis *= 255
            rendered_mesh_texture_vis = rendered_mesh_texture_vis.astype(np.uint8)
            rendered_mesh_texture_vis = cv2.cvtColor(rendered_mesh_texture_vis, cv2.COLOR_GRAY2RGB)
            rendered_mesh_texture_vis = cv2.applyColorMap(rendered_mesh_texture_vis, cv2.COLORMAP_JET)

        rendered_color_overlaid = copy.deepcopy(color_img)

        if not transparent_overlay:
            rendered_color_overlaid[rendered_mesh_mask_vis[:, :, 2] == 1, :] = \
                (
                    rendered_mesh_texture_vis)[rendered_mesh_mask_vis[:, :, 2] == 1, :]
        else:
            rendered_color_overlaid[rendered_mesh_mask_vis[:, :, 2] == 1, :] = \
                color_img[rendered_mesh_mask_vis[:, :, 2] == 1, :] * 0.3 + (
                        (rendered_mesh_texture_vis[rendered_mesh_mask_vis[:, :, 2] == 1, :]) * 0.7)

        vis_img = np.concatenate((color_img, rendered_color_overlaid), axis=1)

        return vis_img
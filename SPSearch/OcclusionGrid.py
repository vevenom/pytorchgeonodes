import torch
import numpy as np
import open3d as o3d

from pytorch3d.ops import sample_points_from_meshes, knn_points
from pytorch3d.loss.chamfer import chamfer_distance

from PytorchGeoNodes.Pytorch3DRenderer.Torch3DRenderer import Torch3DRenderer
from SPSearch.utils import calculate_floor_plane

class OcclusionGrid(torch.nn.Module):
    def __init__(self, surface_points=None, other_surface_points=None):
        super(OcclusionGrid, self).__init__()

        surface_knn = knn_points(other_surface_points[None], surface_points[None], K=1, return_nn=False, norm=1)
        surface_knn_dists = surface_knn.dists[0]
        close_nn = surface_knn_dists <= 0.5
        self.other_surface_points = other_surface_points[close_nn[:,0]]

        self.object_pcd = surface_points
        if surface_points is None:
            self.surface_points_3d = torch.zeros((0,3))
        else:
            self.surface_points_3d = surface_points
        self.occluding_points_3d = torch.zeros((0, 3))
        self.occluding_points_values = torch.zeros((0, 1))
        self.occluded_surface_points = torch.zeros((0, 3))
        self.occluding_points_gradients = torch.zeros((0, 3))
        self.floor_plane = torch.tensor([0,1.,0,0])
        min_y = torch.min(self.other_surface_points[:, 1], dim=0)[0]
        self.floor_plane[-1] = -min_y

        self.torch_renderer = Torch3DRenderer()

        self.epsilon = 1e-3
        self.max_depth = 3.0
        self.voxel_size_surface = 0.01  # 1cm
        self.voxel_size_occluder = 0.05 # 5cm use smaller grid for faster loss calculation

        self.occluder_step_size = self.voxel_size_occluder # 0.05
        self.cd_clamp = 0.1
        self.distance_clamp = 0.3
        self.max_ray_offset = 2.
        self.ray_offsets_positive = torch.arange(self.voxel_size_occluder + 0.01, self.max_ray_offset, self.occluder_step_size)
        self.voting_census = 1

    def to(self, device):
        super(OcclusionGrid, self).to(device)

        self.surface_points_3d = self.surface_points_3d.to(device)
        self.occluding_points_3d = self.occluding_points_3d.to(device)
        self.occluding_points_values = self.occluding_points_values.to(device)
        self.occluded_surface_points = self.occluded_surface_points.to(device)
        self.ray_offsets_positive = self.ray_offsets_positive.to(device)
        self.floor_plane = self.floor_plane.to(device)

    def visualize_loss(self, obj_mesh):

        mesh_pcd = (
            sample_points_from_meshes(obj_mesh, num_samples=10000))

        surface_points_3d_np = self.surface_points_3d.cpu().numpy()
        o3d_surface_points_3d = o3d.geometry.PointCloud()
        o3d_surface_points_3d.points = o3d.utility.Vector3dVector()
        o3d_surface_points_3d.points = o3d.utility.Vector3dVector(surface_points_3d_np)
        o3d_surface_points_3d.colors = o3d.utility.Vector3dVector(np.zeros_like(surface_points_3d_np) + np.array([0, 0.5, 0]))

        occluding_points_3d = self.occluding_points_3d.cpu().numpy()
        occluder_sp_colors = np.ones_like(occluding_points_3d) * np.array([[1., 0, 0]])

        # occluder_sp_colors = occluder_sp_colors * np.clip(occluding_points_3d_values, a_max=0.3, a_min=None) / 0.3
        # occluding_points_3d = occluding_points_3d[occluding_points_3d_values[..., 0] > 0.5, :]
        # occluder_sp_colors = occluder_sp_colors[occluding_points_3d_values[..., 0] > 0.5, :]
        o3d_occluder_sp_pcd = o3d.geometry.PointCloud()
        o3d_occluder_sp_pcd.points = o3d.utility.Vector3dVector(occluding_points_3d)
        o3d_occluder_sp_pcd.colors = o3d.utility.Vector3dVector(occluder_sp_colors)

        o3d_obj_mesh_3d = o3d.geometry.PointCloud()
        o3d_obj_mesh_3d.points = o3d.utility.Vector3dVector(mesh_pcd[0].detach().cpu().numpy())

        o3d.visualization.draw_geometries([o3d_surface_points_3d, o3d_obj_mesh_3d])
        # o3d.visualization.draw_geometries([o3d_surface_points_3d, o3d_occluder_sp_pcd, o3d_obj_mesh_3d])

        return o3d_surface_points_3d

    def calculate_loss_from_mesh(self, obj_mesh):

        mesh_pcd = (
            sample_points_from_meshes(obj_mesh, num_samples=20000))

        grid_points = self.occluding_points_3d

        (cd_loss_x, cd_loss_y) = chamfer_distance(self.surface_points_3d[None], mesh_pcd,
                                                batch_reduction=None,
                                                point_reduction=None,
                                                norm=1,
                                                single_directional=False)[0]

        cd_loss = 1.0 * cd_loss_x.mean()

        mesh_knn = knn_points(mesh_pcd, grid_points[None], K=1, return_nn=False, norm=2)
        mesh_knn_dists = mesh_knn.dists[0]

        is_mesh_on_g = (torch.sqrt(mesh_knn_dists) < 1 * self.voxel_size_occluder)[..., None]
        occlusion_cost2 = is_mesh_on_g * cd_loss_y[None]
        occlusion_cost2 = occlusion_cost2.mean()
        occlusion_cost = occlusion_cost2

        bb = obj_mesh.get_bounding_boxes()  # (N, 3, 2)
        reconstructed_scale = (bb[..., 1] - bb[..., 0]) ** 2
        reconstructed_obj_center = (bb[..., 0] + bb[..., 1]) / 2
        reconstructed_floor_plane = calculate_floor_plane(reconstructed_obj_center, reconstructed_scale)

        support_plane_loss = torch.abs(self.floor_plane[-1] - reconstructed_floor_plane[-1])

        loss = 1.0 * cd_loss + 0.5 * occlusion_cost + 0.01 * support_plane_loss

        return loss

    def calculate_loss_from_sampled_points(self, obj_pcd):

        grid_points = self.occluding_points_3d

        (cd_loss_x, cd_loss_y) = chamfer_distance(self.surface_points_3d[None], obj_pcd,
                                                  batch_reduction=None,
                                                  point_reduction=None,
                                                  norm=1,
                                                  single_directional=False)[0]  # / self.cd_clamp

        cd_loss = 1.0 * cd_loss_x.mean()

        mesh_knn = knn_points(obj_pcd, grid_points[None], K=1, return_nn=False, norm=2)
        mesh_knn_dists = mesh_knn.dists[0]

        is_mesh_on_g = (torch.sqrt(mesh_knn_dists) < 1 * self.voxel_size_occluder)[..., None]
        occlusion_cost2 = is_mesh_on_g * cd_loss_y[None]
        occlusion_cost2 = occlusion_cost2.mean()
        occlusion_cost = occlusion_cost2

        loss = 1.0 * cd_loss + 0.1 * occlusion_cost

        return loss

    def visualize_grid_open3d(self, other_points=None):

        full_colors = np.ones_like(self.surface_points_3d.cpu().numpy()) * np.array([[0, 0.5, 0]])
        o3d_surface_pcd = o3d.geometry.PointCloud()
        o3d_surface_pcd.points = o3d.utility.Vector3dVector(self.surface_points_3d.cpu().numpy())
        o3d_surface_pcd.colors = o3d.utility.Vector3dVector(full_colors)

        occluding_points_3d = self.occluding_points_3d.cpu().numpy()
        occluder_sp_colors = np.ones_like(occluding_points_3d) * np.array([[1., 0, 0]])

        o3d_occluder_sp_pcd = o3d.geometry.PointCloud()
        o3d_occluder_sp_pcd.points = o3d.utility.Vector3dVector(occluding_points_3d)
        o3d_occluder_sp_pcd.colors = o3d.utility.Vector3dVector(occluder_sp_colors)

        full_colors = np.ones_like(self.other_surface_points.cpu().numpy()) * np.array([[0, 0., 0.5]])
        o3d_other_surface_pcd = o3d.geometry.PointCloud()
        o3d_other_surface_pcd.points = o3d.utility.Vector3dVector(self.other_surface_points.cpu().numpy())
        o3d_other_surface_pcd.colors = o3d.utility.Vector3dVector(full_colors)

        occluded_points_3d = self.occluded_surface_points.cpu().numpy()

        print('Number of occluder points: ', occluding_points_3d.shape[0])

        # o3d.visualization.draw_geometries([o3d_surface_pcd, o3d_other_surface_pcd])
        o3d.visualization.draw_geometries([o3d_surface_pcd, o3d_other_surface_pcd,o3d_occluder_sp_pcd])

    def calculate_grid(self, scene_batch):

        # surface_points points
        mask_gt = scene_batch['instance_seg'].bool()
        scene_depth = torch.clone(scene_batch['depth'])
        scene_depth[scene_depth >= self.max_depth] = 0

        # Calculate surface points from 2D instance masks
        new_surface_points_3d, _ = self.calculate_points(scene_batch,
                                                      mask_gt,
                                                      scene_depth,
                                                      scene_depth,
                                                      voxel_size=self.voxel_size_surface)
        surface_points_3d_other_nn = knn_points(new_surface_points_3d[None], self.other_surface_points[None], K=1, norm=1)
        new_surface_points_3d = new_surface_points_3d[surface_points_3d_other_nn.dists[0,...,0] > 0.05, :]

        # Remove noise (points that are far away from the original surface points)
        surface_points_3d_nn = knn_points(new_surface_points_3d[None], self.surface_points_3d[None], K=1, norm=1)
        new_surface_points_3d = new_surface_points_3d[surface_points_3d_nn.dists[0,...,0] < 0.2]

        # Concatenate initial and new surface points
        self.surface_points_3d = torch.cat([self.surface_points_3d, new_surface_points_3d], dim=0)

        self.surface_points_3d = torch.round(self.surface_points_3d / self.voxel_size_surface) * self.voxel_size_surface
        self.surface_points_3d = torch.unique(self.surface_points_3d, dim=0)

        occluder_mask = torch.ones_like(mask_gt)
        ray_offsets = self.ray_offsets_positive

        # Calculate occlusion grid from rays of different views
        camera_plane = 0.0
        for ro_ind in range(ray_offsets.shape[0]):
            ray_offset_depth = torch.clamp_min(scene_depth - ray_offsets[ro_ind], camera_plane)
            scene_depth_masked = scene_depth.clone()

            scene_depth_masked[ray_offset_depth == camera_plane] = 0.
            ray_offset_depth[ray_offset_depth == camera_plane] = 0.

            occluding_points_3d, _ = (
                self.calculate_points(scene_batch,
                                      occluder_mask,
                                      ray_offset_depth,
                                      scene_depth_masked,
                                      voxel_size=self.voxel_size_occluder))

            if occluding_points_3d.shape[0] == 0:
                continue

            self.occluding_points_3d = torch.cat([self.occluding_points_3d,
                                                  occluding_points_3d], dim=0)

        self.occluding_points_3d = torch.unique(self.occluding_points_3d, dim=0)

        comparison_points = new_surface_points_3d
        occl_knn = knn_points(self.occluding_points_3d[None], comparison_points[None], K=1, return_nn=False)
        non_occluding_nn_idx = occl_knn.idx[0]
        non_occluding_nn_points = comparison_points[non_occluding_nn_idx.view(-1), :]
        occluded_surface_points = non_occluding_nn_points

        self.occluded_surface_points = occluded_surface_points

    def calculate_points(self, scene_batch, masks, occluder_depths, occluded_depths, voxel_size):

            device = self.occluding_points_3d.device

            color_img = scene_batch['color'] / 255.0

            poses = scene_batch['pose']

            intrinsics = scene_batch['intrinsics']

            image_size = color_img[0].shape[:2]
            image_size = (image_size[1], image_size[0])

            x_arange = torch.arange(0, image_size[0], device=device)
            y_arange = torch.arange(0, image_size[1], device=device)
            ys, xs = torch.meshgrid([y_arange, x_arange], indexing='ij')

            points_6d = torch.zeros((0, 6), device=device)

            def project_image23d(pose, depth_map, mask, intrinsics_frame, ys, xs, voxel_size):
                intrinsics_inv = torch.linalg.inv(intrinsics_frame)

                zs = depth_map[ys, xs]
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

                return xyz_global

            for frame_ind in range(poses.shape[0]):

                pose = poses[frame_ind]
                depth_map = occluder_depths[frame_ind]
                occluded_depth_map = occluded_depths[frame_ind]
                mask = masks[frame_ind]

                mask = mask.view(-1)
                intrinsics_frame = intrinsics[frame_ind][:3, :3].to(torch.float32)

                xyz_global = project_image23d(pose, depth_map, mask, intrinsics_frame, ys, xs,
                                              voxel_size)
                xyz_global_occluded = project_image23d(pose, occluded_depth_map, mask, intrinsics_frame, ys, xs,
                                                       voxel_size)

                xyz_global = torch.cat([xyz_global, xyz_global_occluded], dim=1)

                xyz_global, xyz_global_object_counts = torch.unique(xyz_global, return_counts=True, dim=0)
                points_6d = torch.concatenate([points_6d, xyz_global],
                                                     dim=0)

            points_6d, points_3d_votes = torch.unique(points_6d, return_counts=True, dim=0)
            points_3d_census = torch.argwhere(points_3d_votes >= self.voting_census)[:,0]
            points_6d = points_6d[points_3d_census, :]

            points_3d = points_6d[:, :3]
            occluded_points_3d = points_6d[:, 3:]

            return points_3d, occluded_points_3d

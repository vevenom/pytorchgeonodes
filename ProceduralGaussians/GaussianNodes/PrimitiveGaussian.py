import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_quaternion, Transform3d
from typing import List
import numpy as np
from scipy.spatial import cKDTree

from .utils import rgb_to_sh, get_triangle_bary_coords

from PytorchGeoNodes.Nodes.PrimitiveMesh import PrimitiveMesh
from PytorchGeoNodes.Nodes.PrimitiveGeometry import PrimitiveGeometry
from PytorchGeoNodes.Nodes.PrimitiveMesh import join_meshes


class PrimitiveGaussian(PrimitiveGeometry):
    def __init__(self, identifier,
                 primitive_type, primitive_mesh: PrimitiveMesh, config,
                 device=None, is_trainable=False):
        torch.nn.Module.__init__(self)
        PrimitiveGeometry.__init__(self, identifier, primitive_type, device)

        self.primitive_type = primitive_type
        self.is_trainable = is_trainable

        self.primitive_mesh = primitive_mesh
        self.config = config

        # Define barycentric coordinates
        self.surface_triangle_bary_coords = get_triangle_bary_coords(
            degree=self.config.gaussian_primitive.n_gaussians_per_surface_triangle)[..., None]

        # Create trainable parameters
        self.params, self.transform_matrices = self.create_params(is_trainable)

    def to(self, device):
        torch.nn.Module.to(self, device)
        self.params.to(device)
        self.primitive_mesh.to(device)
        self.surface_triangle_bary_coords.to(device)
        self.transform_matrices = self.transform_matrices.to(device)

    def create_params(self, is_trainable):
        """
        Create trainable parameters for the primitive.
        """
        verts, faces = self.primitive_mesh.verts, self.primitive_mesh.faces
        device = verts.device
        B, M, _ = faces.shape
        config = self.config.gaussian_primitive

        total_n_gaussians = M * config.n_gaussians_per_surface_triangle

        verts_offsets = torch.zeros((1, verts.shape[1], 3), device=device)
        means = torch.zeros((1, total_n_gaussians, 3), device=device)

        verts_colors = torch.zeros((1, verts_offsets.shape[1], 3), device=device)
        scales = torch.zeros((1, total_n_gaussians, 2), device=device) + 1e-3
        complex_number = torch.zeros((1, total_n_gaussians, 2), device=device)

        if config.spherical_harmonics_degree is None:
            colors = torch.zeros((1, total_n_gaussians, 3), device=device)
        elif config.spherical_harmonics_degree > 0:
            colors = torch.zeros((
                1, total_n_gaussians, (config.spherical_harmonics_degree + 1) ** 2, 3))  # [1, N, K, 3]
        else:
            assert False, "Spherical harmonics degree has to be > 0"

        opacities = torch.zeros((1, total_n_gaussians, 1), device=device) + config.init_opacity
        transform_matrices = torch.eye(4, device=device)[None, None]

        scale_thickness = torch.zeros_like(scales[..., :1])

        if is_trainable:
            params_dict = {
                'verts_offsets': nn.ParameterList([nn.Parameter(
                    verts_offsets, requires_grad=config.optimize_verts_offsets)]),
                'verts_colors_offsets': nn.ParameterList([nn.Parameter(
                    verts_colors, requires_grad=config.optimize_verts_colors_offsets)]),
                'means_offsets': nn.ParameterList([nn.Parameter(
                    means, requires_grad=config.optimize_means_offsets)]),
                'scales_offsets': nn.ParameterList([nn.Parameter(
                    scales, requires_grad=config.optimize_scales_offsets)]),
                'scale_thickness_offsets': nn.ParameterList([nn.Parameter(
                    scale_thickness, requires_grad=config.optimize_scale_thickness_offsets)]),
                'complex_offsets': nn.ParameterList([nn.Parameter(
                    complex_number, requires_grad=config.optimize_complex_offsets)]),
                'colors_offsets': nn.ParameterList([nn.Parameter(
                    colors, requires_grad=config.optimize_colors_offsets)]),
                'opacities_offsets': nn.ParameterList([nn.Parameter(opacities,
                                                                    requires_grad=config.optimize_opacities_offsets)]),
            }
        else:
            params_dict = {
                'verts_offsets': nn.ParameterList([nn.Parameter(verts_offsets, requires_grad=False)]),
                'verts_colors_offsets': nn.ParameterList([nn.Parameter(verts_colors, requires_grad=False)]),
                'means_offsets': nn.ParameterList([nn.Parameter(means, requires_grad=False)]),
                'scales_offsets': nn.ParameterList([nn.Parameter(scales, requires_grad=False)]),
                'scale_thickness_offsets': nn.ParameterList([nn.Parameter(scale_thickness, requires_grad=False)]),
                'complex_offsets': nn.ParameterList([nn.Parameter(complex_number, requires_grad=False)]),
                'colors_offsets': nn.ParameterList([nn.Parameter(colors, requires_grad=False)]),
                'opacities_offsets': nn.ParameterList([nn.Parameter(opacities, requires_grad=False)])
            }

        return nn.ParameterDict(params_dict), transform_matrices

    @staticmethod
    def create_empty(device):
        primitive_mesh = PrimitiveMesh('Empty', device=device)

        return PrimitiveGaussian('Empty', primitive_mesh, is_trainable=False, device=device)

    def get_num_gaussians(self) -> int:
        return sum([m_offs.shape[1] for m_offs in self.params['means_offsets']])

    def is_empty(self):
        return self.get_num_gaussians() == 0

    @property
    def batch_size(self):
        return self.params['means_offsets'][0].shape[0]

    def get_device(self):
        return self.params['means_offsets'][0].device

    def scale(self, size):
        self.primitive_mesh.scale(size)

        scale_matrix = torch.eye(4, device=self.get_device())[None, None]
        scale_matrix[..., 0, 0] = size[0, 0, 0]
        scale_matrix[..., 1, 1] = size[0, 0, 1]
        scale_matrix[..., 2, 2] = size[0, 0, 2]

        self.transform_matrices = torch.matmul(self.transform_matrices, scale_matrix)

    def transform(self, transform3d):
        pg = self.clone()

        pg.primitive_mesh = pg.primitive_mesh.transform(transform3d)
        pg.transform_matrices = torch.matmul(pg.transform_matrices, transform3d.get_matrix()[None])

        return pg

    def clone(self):
        """
        Create a copy of the primitive geometry. Parameters are shared between clones.
        """
        cloned_primitive_mesh = self.primitive_mesh.clone()
        new_primitive = PrimitiveGaussian(-1,
                                          self.primitive_type, cloned_primitive_mesh,
                                          config=self.config, device=self.get_device())
        new_primitive.params = self.params
        new_primitive.transform_matrices = self.transform_matrices

        return new_primitive

    def get_verts_offsets(self, use_transform=True):
        """
        Get verts offsets for primitive geometry and optionally apply corresponding affine transformations.
        """
        v_offs_list = []
        curr_offset = 0
        for param_ind, v_offs in enumerate(self.params["verts_offsets"]):
            rotate_scale_matrix = self.transform_matrices[:, param_ind].clone()
            rotate_scale_matrix[..., 3, :3] = 0
            transform3d = Transform3d(matrix=rotate_scale_matrix)

            if use_transform:
                v_offs = transform3d.transform_points(v_offs)
            v_offs_list.append(v_offs)

            curr_offset += v_offs.shape[1]
        verts_offsets = torch.cat(v_offs_list, dim=1)

        return verts_offsets

    def compute_final_verts(self):
        verts_offsets = self.get_verts_offsets(use_transform=True)

        verts = self.primitive_mesh.verts + verts_offsets

        return verts

    def get_means_offsets(self, use_transform=True):
        """
        Get means offsets for primitive geometry and optionally apply corresponding affine transformations.
        """
        m_offs_list = []
        curr_offset = 0
        for param_ind, m_offs in enumerate(self.params["means_offsets"]):
            rotate_scale_matrix = self.transform_matrices[:, param_ind].clone()
            rotate_scale_matrix[..., 3, :3] = 0
            transform3d = Transform3d(matrix=rotate_scale_matrix)

            if use_transform:
                m_offs = transform3d.transform_points(m_offs)
            m_offs_list.append(m_offs)

            curr_offset += m_offs.shape[1]

        means_offsets = torch.cat(m_offs_list, dim=1)

        return means_offsets

    def compute_final_means(self, means) -> torch.Tensor:
        means_offsets = self.get_means_offsets(use_transform=True)

        means = means + means_offsets

        return means

    def compute_final_scales(self, scales) -> torch.Tensor:
        """
        Compute final scales for primitive geometry.
        """
        config = self.config.gaussian_primitive

        s_offs_list = []
        for param_ind, s_offs in enumerate(self.params["scales_offsets"]):
            rotate_scale_matrix = self.transform_matrices[:, param_ind]

            scales_coeffs = torch.linalg.norm(rotate_scale_matrix[..., :3, :3], dim=-2, keepdim=True)

            scale_matrix = torch.eye(4, device=self.get_device())[None]
            scale_matrix[..., 0, 0] = scales_coeffs[...,0]
            scale_matrix[..., 1, 1] = scales_coeffs[...,1]
            scale_matrix[..., 2, 2] = scales_coeffs[...,2]

            st_offs = self.params["scale_thickness_offsets"][param_ind]
            s_offs = torch.cat([st_offs, s_offs], dim=-1)

            transform3d = Transform3d(matrix=scale_matrix)
            s_offs_list.append(transform3d.transform_points(s_offs))

        scales_offsets = torch.cat(s_offs_list, dim=1)
        scales_offs_norm = torch.sigmoid(scales_offsets[..., 1:]) * 2 - 1 #[-1,1]
        scales = scales + scales * scales_offs_norm

        if not config.flat_gaussians:
            scales = torch.cat([
                scales_offsets[..., :1],
                scales,
            ], dim=-1)  # [1, total_n_gaussians, 3]
        else:
            scales = torch.cat([
                config.init_gaussian_thickness * torch.ones_like(scales_offsets[..., :1]),
                scales,
            ], dim=-1)  # [1, total_n_gaussians, 3]

        return scales

    def compute_final_colors(self) -> torch.Tensor:
        colors_offsets = torch.cat([c_offs for c_offs in self.params["colors_offsets"]], dim=1)

        colors = colors_offsets

        return colors

    def compute_final_opacities(self) -> torch.Tensor:
        opacities_offsets = torch.cat([o_offs for o_offs in self.params["opacities_offsets"]], dim=1)

        return opacities_offsets

    def get_mesh(self, use_offsets=True, return_verts_faces=False):
        if not use_offsets:
            verts, faces = self.primitive_mesh.verts, self.primitive_mesh.faces
        else:
            faces = self.primitive_mesh.faces
            verts = self.compute_final_verts()

        mesh_py3d = Meshes(verts=[verts[0]], faces=[faces[0]])

        if return_verts_faces:
            return mesh_py3d, verts, faces
        return mesh_py3d

    def get_final_quats(self) -> torch.Tensor:
        """
        Get rotation quaternions based on triangle orientation.
        """
        config = self.config.gaussian_primitive

        # _, faces = self.primitive_mesh.verts, self.primitive_mesh.faces
        mesh_py3d, verts, faces = self.get_mesh(return_verts_faces=True)

        R_0 = torch.nn.functional.normalize(mesh_py3d.faces_normals_list()[0], dim=-1).unsqueeze(0)  # [1, M, 3]
        # print("R_0: ", R_0)
        # We use the first side of every triangle as the second base axis
        faces_verts = verts[:, faces].squeeze(1)  # [1, M, 3, 3]
        base_R_1 = torch.nn.functional.normalize(faces_verts[:, :, 0] - faces_verts[:, :, 1], dim=-1)  # [1, M, 3]
        base_R_2 = torch.nn.functional.normalize(torch.cross(R_0, base_R_1, dim=-1), dim=-1)  # [1, M, 3]

        complex_numbers = torch.cat([cn_offs for cn_offs in self.params["complex_offsets"]], dim=1)

        complex_numbers[..., 0] += 1
        complex_numbers = torch.nn.functional.normalize(complex_numbers, dim=-1)
        complex_numbers = complex_numbers.view(1,
                                               faces.shape[1],
                                               config.n_gaussians_per_surface_triangle, 2)  # [1, M, 4, 2]

        R_1 = complex_numbers[..., 0:1] * base_R_1[:, :, None] + complex_numbers[..., 1:2] * base_R_2[:, :,
                                                                                             None]  # [1, M, 4, 3]
        R_2 = -complex_numbers[..., 1:2] * base_R_1[:, :, None] + complex_numbers[..., 0:1] * base_R_2[:, :,
                                                                                              None]  # [1, M, 4, 3]

        R = torch.cat([R_0[:, :, None, ..., None].expand(
            -1, -1, config.n_gaussians_per_surface_triangle, -1, -1).clone(),
                       R_1[..., None],
                       R_2[..., None]],
                      dim=-1).view(1, -1, 3, 3)  # [1, total_n_gaussians, 3, 3]

        quaternion = matrix_to_quaternion(R)  # [1, total_n_gaussians, 4]

        return torch.nn.functional.normalize(quaternion, dim=-1)

    def assign_colors(self, colors_init):
        config = self.config.gaussian_primitive

        colors_offs = 0
        for c_id, c_offs in enumerate(self.params["colors_offsets"]):
            if config.spherical_harmonics_degree is None:
                c_offs.data = colors_init[:, colors_offs:colors_offs + c_offs.shape[1]]
                colors_offs += c_offs.shape[1]
            elif config.spherical_harmonics_degree > 0:
                c_offs.data[:,:,1] = rgb_to_sh(colors_init[:, colors_offs:colors_offs + c_offs.shape[1]])
                colors_offs += c_offs.shape[1]
            else:
                assert False, "Spherical harmonics degree has to be > 0"

    def compute_gaussians(self):
        init_gaussians = self.initialize_gaussians()
        final_gaussians = self.compute_final_gaussians(init_gaussians)

        return init_gaussians, final_gaussians

    def compute_final_gaussians(self, initial_gaussians):
        means = self.compute_final_means(initial_gaussians['means'])

        colors = self.compute_final_colors()
        opacities = self.compute_final_opacities()
        scales = self.compute_final_scales(initial_gaussians['scales'])

        quats = self.get_final_quats()

        gaussians = {'means': means, 'scales': scales, 'quats': quats,
                     'colors': colors, 'opacities': opacities}

        return gaussians

    def initialize_gaussians(self):
        _, faces = self.primitive_mesh.verts, self.primitive_mesh.faces
        verts = self.compute_final_verts()

        B, M, _ = faces.shape
        config = self.config.gaussian_primitive

        total_n_gaussians = M * config.n_gaussians_per_surface_triangle
        faces_verts = verts[:, faces].squeeze(1)  # [1, M, 3, 3]

        means = faces_verts[:, :, None] * self.surface_triangle_bary_coords[
            None, None].to(faces_verts.device)  # [1, M, 1, 3, 3] * [1, 1, 4, 3, 1] = [1, M, 4, 3, 3]
        means = means.sum(dim=-2)  # [1, M, 4, 3]
        means = means.reshape(B, -1, 3)  # [B, n_gaussians, 3] --> n_gaussians = M * n_gaussians_per_face

        # From face size
        scales = (faces_verts - faces_verts[:, :, [1, 2, 0]]).norm(dim=-1).max(dim=-1)[
                     # NOTE: Should I use max or min ???
                     0] * config.surface_triangle_circle_radius  # [1, M]
        scales = scales[..., None, None].expand(
            -1, -1,
            config.n_gaussians_per_surface_triangle,
            2).reshape(1, -1, 2)  # [1, total_n_gaussians, 2]

        gaussians = {'means': means, 'scales': scales}

        return gaussians

    def initialize_from_colored_pcd(self, xyz_colors):

        gaussians = self.initialize_gaussians()
        means = gaussians['means']
        scales = gaussians['scales']

        xyz_colors_np = xyz_colors.detach().cpu().numpy()
        sampled_xyz = np.asarray(xyz_colors_np[:, :3])
        sampled_colors = np.asarray(xyz_colors_np[:, 3:])
        means_np = means.clone().detach().cpu().numpy()

        kdtree = cKDTree(sampled_xyz)
        distances, indices = kdtree.query(means_np, k=1)
        colors_init = torch.tensor(sampled_colors[indices], dtype=torch.float32).to(self.get_device())

        gaussians['colors'] = colors_init

        return gaussians  # means, scales, quats, colors, opacities

    def export_to_file(self, save_path):
        ckpt = {
            'params': self.params,
            'initial_gaussians': self.init_gaussians
        }
        assert False
        torch.save(ckpt, save_path)

    @staticmethod
    def assign_from_file(self, load_path):
        ckpt = torch.load(load_path)
        self.params = ckpt["params"]
        self.initial_gaussians = ckpt["initial_gaussians"]


def join_geometries(primitive_gaussians: List[PrimitiveGaussian]):
    primitive_gaussians = [pg for pg in primitive_gaussians if not pg.is_empty()]  # Ignore empty primitives
    device = primitive_gaussians[0].get_device()

    if not len(primitive_gaussians):
        return PrimitiveGaussian('Empty', PrimitiveMesh('Empty'),
                                 config=None, device=device)

    total_num_gaussians = sum(mesh.get_num_gaussians() for mesh in primitive_gaussians)
    curr_num_gaussians = 0

    joined_verts = torch.nn.ParameterList()
    joined_verts_colors = torch.nn.ParameterList()

    joined_means = torch.nn.ParameterList()

    joined_scales = torch.nn.ParameterList()
    joined_scales_thickness = torch.nn.ParameterList()
    joined_quats = torch.nn.ParameterList()
    joined_colors = torch.nn.ParameterList()
    joined_opacities = torch.nn.ParameterList()
    joined_transform_matrices = torch.zeros((1, 0, 4, 4), device=device)

    for pg in primitive_gaussians:
        if pg.is_empty():
            continue
        joined_verts.extend(pg.params['verts_offsets'])
        joined_verts_colors.extend(pg.params['verts_colors_offsets'])
        joined_means.extend(pg.params['means_offsets'])
        joined_scales.extend(pg.params['scales_offsets'])
        joined_scales_thickness.extend(pg.params['scale_thickness_offsets'])
        joined_quats.extend(pg.params['complex_offsets'])
        joined_colors.extend(pg.params['colors_offsets'])
        joined_opacities.extend(pg.params['opacities_offsets'])
        joined_transform_matrices = torch.cat([joined_transform_matrices, pg.transform_matrices], dim=1)
        curr_num_gaussians += pg.get_num_gaussians()

    primitive_meshes = [pg.primitive_mesh.clone() for pg in primitive_gaussians]
    new_mesh = join_meshes(primitive_meshes)

    config = primitive_gaussians[0].config
    new_primitive = PrimitiveGaussian(-1, 'Joined', new_mesh,
                                      config=config, device=device)
    new_primitive.params['verts_offsets'] = joined_verts
    new_primitive.params['verts_colors_offsets'] = joined_verts_colors
    new_primitive.params['means_offsets'] = joined_means
    new_primitive.params['scales_offsets'] = joined_scales
    new_primitive.params['scale_thickness_offsets'] = joined_scales_thickness
    new_primitive.params['complex_offsets'] = joined_quats
    new_primitive.params['colors_offsets'] = joined_colors
    new_primitive.params['opacities_offsets'] = joined_opacities

    new_primitive.transform_matrices = joined_transform_matrices
    new_primitive.total_n_gaussians = curr_num_gaussians

    return new_primitive

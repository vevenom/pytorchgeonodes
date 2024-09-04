import bpy
import torch
import numpy as np
import copy

from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Translate, Scale, Transform3d, euler_angles_to_matrix

from PytorchGeoNodes.Nodes.Node import *


class NodeTransformGeometryStrings:
    Geometry_str = 'Geometry'
    Translation_str = 'Translation'
    Rotation_str = 'Rotation'
    Scale_str = 'Scale'


class NodeTransformGeometry(Node):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Transform Geometry Node allows you to move, rotate or scale the geometry. The transformation is applied to
        the entire geometry, and not per element. The Set Position Node is used for moving individual points of a
        geometry. For transforming instances individually, the instance translate, rotate, or scale nodes can be used.

        :param bpy_node:
        """
        super().__init__(bpy_node)
        print('Creating NodeTransformGeometry')

        self.cached_output = None

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        geometry = inputs_dict[self.name][NodeStrings.IN_str +
                                          NodeTransformGeometryStrings.Geometry_str] # should be a list of mesh
        translation = inputs_dict[self.name][NodeStrings.IN_str +
                                             NodeTransformGeometryStrings.Translation_str]
        translation = torch.zeros_like(translation) - translation
        translation[..., 0] = -translation[..., 0]  # invert y and z axis

        rotation = inputs_dict[self.name][NodeStrings.IN_str +
                                          NodeTransformGeometryStrings.Rotation_str]
        scale = inputs_dict[self.name][NodeStrings.IN_str +
                                       NodeTransformGeometryStrings.Scale_str]

        transformed_meshes = []
        for mesh_i, mesh in enumerate(geometry):
            assert isinstance(mesh, Meshes), 'Geometry must be a mesh'
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            verts = mesh.verts_packed()
            faces = mesh.faces_packed()

            translation_i = translation[mesh_i]

            # rotation_i is in Euler angles, convert to rotation matrix
            rotation_i = euler_angles_to_matrix(rotation[mesh_i], convention='XYZ')
            scale_i = scale[mesh_i]

            verts = verts[None].expand(1, verts.shape[0], verts.shape[1])

            # create transform matrix from translation, rotation and scale
            transform_matrix = torch.eye(4, device=self.device)
            transform_matrix = transform_matrix[None].expand(1, transform_matrix.shape[0], transform_matrix.shape[1])
            transform_matrix[:, 3, :3] = translation_i
            transform_matrix[:, :3, :3] = rotation_i

            scale_mat = torch.eye(4, device=self.device)
            scale_mat = scale_mat.view(1, 4, 4).expand(1, 4, 4)
            scale_mat[:, 0, 0] = scale_i[:, 0]
            scale_mat[:, 1, 1] = scale_i[:, 1]
            scale_mat[:, 2, 1] = scale_i[:, 2]

            scale = Scale(scale_i)
            transform = scale.compose(Transform3d(matrix=transform_matrix))

            verts = transform.transform_points(verts)[0]

            transformed_mesh = Meshes(verts=[verts], faces=[faces])

            transformed_meshes.append(transformed_mesh)

        inputs_dict[self.name][NodeStrings.OUT_str + 'Geometry'] = transformed_meshes


import torch
import numpy as np
import os

from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Translate, Scale
from pytorch3d.transforms import Transform3d
from pytorch3d.io import load_obj

from PytorchGeoNodes.Nodes.Node import *
from PytorchGeoNodes.Nodes.PrimitiveMesh import PrimitiveMesh


class NodeMeshPrimitiveCylinderStrings:
    Vertices_str = 'Vertices'
    Side_Segments_str = 'Side Segments'
    Fill_Segments_str = 'Fill Segments'
    Radius_str = 'Radius'
    Depth_str = 'Depth'


class PrimitiveCylinder(PrimitiveMesh):
    def __init__(self, primitive_id=-1, device=None):
        super().__init__(primitive_id, primitive_type='Cylinder')

        verts, faces = PrimitiveCylinder.generate_unit_cylinder_mesh(device=device)
        self.set_all(verts, faces, base_id=primitive_id)

    @staticmethod
    def generate_unit_cylinder_mesh(device):
        # .obj file of cylinder is located in the same directory as this file in primitives/cylinder.obj
        current_directory = os.path.dirname(os.path.realpath(__file__))
        cylinder_obj_path = os.path.join(current_directory, 'primitives/cylinder.obj')

        # Transform the primitive from blender to pytorch3d
        transform_py3d2blender = np.eye(4)
        transform_py3d2blender[0, 0] = 1
        transform_py3d2blender[1, 1] = 0
        transform_py3d2blender[1, 2] = -1
        transform_py3d2blender[2, 1] = 1
        transform_py3d2blender[2, 2] = 0
        # transform_py3d2blender = np.linalg.inv(transform_py3d2blender)
        transform_py3d2blender = torch.tensor(transform_py3d2blender, dtype=torch.float32)
        transform_blender2py3d = Transform3d(matrix=transform_py3d2blender)

        with open(cylinder_obj_path, 'r') as f:
            verts, faces, _ = load_obj(f, load_textures=False, device=device)
        verts = transform_blender2py3d.transform_points(verts)
        # the .obj primitive is twice in size compared to the one in the graph
        verts *= 0.5

        return verts[None], faces.verts_idx[None]


class NodePrimitiveCylinder(Node):
    def __init__(self, primitive_id, bpy_node, config):
        """
        The Cylinder node generates a cylinder mesh. It is similar to the Cone node but always uses the same radius
        for the circles at the top and bottom.

        Inputs:
            -- Counts: Number of vertices on the line.
            -- Start Location: Position of the first vertex. (X, Y, Z)
            -- Offset: Offset of the vertices along the line. (X, Y, Z)

        Outputs: Vector

        :param bpy_node:
        """
        super().__init__(bpy_node, config)
        print('Creating MeshPrimitiveCylinder')

        self.default_cylinder = PrimitiveCylinder(primitive_id, self.device)

        self.cached_output = None
        
    def to(self, device):
        super().to(device)
        self.default_cylinder.to(device)

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        if not len(self.in_edges) and self.cached_output is not None:
            inputs_dict[self.name][NodeStrings.OUT_str + 'Mesh'] = \
                self.cached_output[NodeStrings.OUT_str + 'Mesh']
            return

        assert torch.all(inputs_dict[self.name][NodeStrings.IN_str +
                                                NodeMeshPrimitiveCylinderStrings.Vertices_str] == 32)
        assert torch.all(inputs_dict[self.name][NodeStrings.IN_str +
                                                NodeMeshPrimitiveCylinderStrings.Side_Segments_str] == 1)
        assert torch.all(inputs_dict[self.name][NodeStrings.IN_str +
                                                NodeMeshPrimitiveCylinderStrings.Fill_Segments_str] == 1)

        cylinder_primitive = self.default_cylinder.clone()

        # radius * 2, depth
        radius2 = 2 * \
                  inputs_dict[self.name][NodeStrings.IN_str +
                                         NodeMeshPrimitiveCylinderStrings.Radius_str][:, None]
        depth = inputs_dict[self.name][NodeStrings.IN_str +
                                       NodeMeshPrimitiveCylinderStrings.Depth_str][:, None]
        size = torch.tensor([[[radius2, radius2, depth]]], device=cylinder_primitive.get_device())
        cylinder_primitive.scale(size)

        inputs_dict[self.name][NodeStrings.OUT_str + 'Mesh'] = [cylinder_primitive]

        if not len(self.in_edges):
            self.cached_output = {NodeStrings.OUT_str + 'Mesh': [cylinder_primitive]}

import bpy
import torch
import numpy as np
import os

from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Translate, Scale
from pytorch3d.io import load_obj

from PytorchGeoNodes.Nodes.Node import *


class NodeMeshPrimitiveCylinderStrings:
    Vertices_str = 'Vertices'
    Side_Segments_str = 'Side Segments'
    Fill_Segments_str = 'Fill Segments'
    Radius_str = 'Radius'
    Depth_str = 'Depth'


def generate_unit_cylinder_mesh(device):

    # .obj file of cylinder is located in the same directory as this file in primitives/cylinder.obj
    current_directory = os.path.dirname(os.path.realpath(__file__))
    cylinder_obj_path = os.path.join(current_directory, 'primitives/cylinder.obj')

    with open(cylinder_obj_path, 'r') as f:
        verts, faces, _ = load_obj(f, load_textures=False, device=device)

    mesh = Meshes(verts=[verts], faces=[faces.verts_idx])

    return mesh


class NodeMeshPrimitiveCylinder(Node):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
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
        super().__init__(bpy_node)
        print('Creating MeshPrimitiveCylinder')

        self.default_cylinder = generate_unit_cylinder_mesh(self.device)

        self.cached_output = None
        
    def to(self, device):
        super().to(device)
        self.default_cylinder = self.default_cylinder.to(device)

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

        # radius * 2, depth

        batch_size = inputs_dict[self.name][NodeStrings.IN_str +
                                            NodeMeshPrimitiveCylinderStrings.Radius_str].shape[0]

        verts = self.default_cylinder.verts_packed()[None]
        verts = verts.expand(batch_size, verts.shape[1], verts.shape[2])

        verts[..., :2] = verts[..., :2] * 2 * \
                         inputs_dict[self.name][NodeStrings.IN_str +
                                                NodeMeshPrimitiveCylinderStrings.Radius_str][:, None]
        verts[..., 2] = verts[..., 2] * \
                        inputs_dict[self.name][NodeStrings.IN_str +
                                               NodeMeshPrimitiveCylinderStrings.Depth_str][:, None]

        faces = self.default_cylinder.faces_packed()
        meshes = []

        for i in range(inputs_dict[self.name][NodeStrings.IN_str +
                                              NodeMeshPrimitiveCylinderStrings.Radius_str].shape[0]):
            meshes.append(Meshes(verts=[verts[i]], faces=[faces]))

        inputs_dict[self.name][NodeStrings.OUT_str + 'Mesh'] = meshes

        if not len(self.in_edges):
            self.cached_output = {NodeStrings.OUT_str + 'Mesh': meshes}

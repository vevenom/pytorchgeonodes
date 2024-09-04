import bpy
import torch
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Translate, Scale

from PytorchGeoNodes.Nodes.Node import *

class NodeMeshPrimitiveCubeStrings:
    Size_str = 'Size'
    Vertices_X_str = 'Vertices X'
    Vertices_Y_str = 'Vertices Y'
    Vertices_Z_str = 'Vertices Z'


def generate_unit_cube_mesh():
    verts = torch.tensor(
        [ # 8 vertices
            [0, 0, 0], # 0
            [0, 0, 1], # 1
            [0, 1, 0], # 2
            [0, 1, 1], # 3
            [1, 0, 0], # 4
            [1, 0, 1], # 5
            [1, 1, 0], # 6
            [1, 1, 1], # 7
        ],
        dtype=torch.float32)
    faces = torch.tensor(
        [ # 12 faces
            [0, 1, 2], # 0
            [1, 3, 2], # 1
            [4, 6, 5], # 2
            [5, 6, 7], # 3
            [0, 4, 5], # 4
            [0, 5, 1], # 5
            [2, 3, 6], # 6
            [3, 7, 6], # 7
            [0, 2, 4], # 8
            [2, 6, 4], # 9
            [1, 5, 3], # 10
            [3, 5, 7], # 11
        ], dtype=torch.int64)

    return Meshes(verts=[verts], faces=[faces])


class NodeMeshPrimitiveCube(Node):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Cube node generates a cuboid mesh with variable side lengths and subdivisions. The inside of the mesh is
        still hollow like a normal cube.

        Inputs:
            -- Counts: Number of vertices on the line.
            -- Start Location: Position of the first vertex. (X, Y, Z)
            -- Offset: Offset of the vertices along the line. (X, Y, Z)

        Outputs: Vector

        :param bpy_node:
        """
        super().__init__(bpy_node)
        print('Creating MeshPrimitiveCube')

        self.default_cube_mesh = generate_unit_cube_mesh().to(self.device)

        self.cached_output = None

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        if not len(self.in_edges) and self.cached_output is not None:
            inputs_dict[self.name][NodeStrings.OUT_str + 'Mesh'] = \
                self.cached_output[NodeStrings.OUT_str + 'Mesh']
            return

        assert torch.all(inputs_dict[self.name][NodeStrings.IN_str +
                                                NodeMeshPrimitiveCubeStrings.Vertices_X_str] == 2)
        assert torch.all(inputs_dict[self.name][NodeStrings.IN_str +
                                                NodeMeshPrimitiveCubeStrings.Vertices_Y_str] == 2)
        assert torch.all(inputs_dict[self.name][NodeStrings.IN_str +
                                                NodeMeshPrimitiveCubeStrings.Vertices_Z_str] == 2)

        size = inputs_dict[self.name][NodeStrings.IN_str +
                                      NodeMeshPrimitiveCubeStrings.Size_str][:, 0]
        cube_mesh = self.default_cube_mesh.to(self.device)

        # Meshes are listed instead of batched. This is because they can be of different shapes in general.
        meshes = []
        verts = cube_mesh.verts_packed()
        faces = cube_mesh.faces_packed()

        # move verts to -0.5 to 0.5
        verts = verts - 0.5

        for i in range(size.shape[0]):
            verts_i = Scale(size[i][None]).transform_points(verts[None])[0]

            meshes.append(Meshes(verts=[verts_i], faces=[faces]))

        inputs_dict[self.name][NodeStrings.OUT_str + 'Mesh'] = meshes

        if not len(self.in_edges):
            self.cached_output = {NodeStrings.OUT_str + 'Mesh': meshes}

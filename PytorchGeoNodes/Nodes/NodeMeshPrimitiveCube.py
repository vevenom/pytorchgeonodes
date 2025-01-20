import torch
import numpy as np
import os

from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Translate, Scale
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.io import load_obj

from PytorchGeoNodes.Nodes.Node import *
from PytorchGeoNodes.Nodes.PrimitiveMesh import PrimitiveMesh

class NodeMeshPrimitiveCubeStrings:
    Size_str = 'Size'
    Vertices_X_str = 'Vertices X'
    Vertices_Y_str = 'Vertices Y'
    Vertices_Z_str = 'Vertices Z'


class PrimitiveCube(PrimitiveMesh):
    def __init__(self, identifier, primitive_filename=None, device=None):
        super().__init__(identifier, primitive_type='Cube')

        if primitive_filename is None:
            verts, faces = PrimitiveCube.generate_unit_cube_mesh(device=device)
            self.set_all(verts, faces, base_id=identifier)
        else:
            verts, faces = self.generate_unit_cube_mesh_from_file(primitive_filename, device=device)
            self.set_all(verts, faces, base_id=identifier)

    @staticmethod
    def generate_unit_cube_mesh(device=None, subdivisions=2):
        verts = torch.tensor(
            [[  # 8 vertices
                [0, 0, 0],  # 0
                [0, 0, 1],  # 1
                [0, 1, 0],  # 2
                [0, 1, 1],  # 3
                [1, 0, 0],  # 4
                [1, 0, 1],  # 5
                [1, 1, 0],  # 6
                [1, 1, 1],  # 7
            ]],
            dtype=torch.float32, device=device)
        verts = verts - 0.5

        faces = torch.tensor(
            [[  # 12 faces
                [0, 1, 2],  # 0
                [1, 3, 2],  # 1
                [4, 6, 5],  # 2
                [5, 6, 7],  # 3
                [0, 4, 5],  # 4
                [0, 5, 1],  # 5
                [2, 3, 6],  # 6
                [3, 7, 6],  # 7
                [0, 2, 4],  # 8
                [2, 6, 4],  # 9
                [1, 5, 3],  # 10
                [3, 5, 7],  # 11
            ]], dtype=torch.int64, device=device)

        for i in range(subdivisions):
            cube_mesh = Meshes(verts=verts, faces=faces)

            cube_mesh = SubdivideMeshes()(cube_mesh)

            verts = cube_mesh.verts_packed()[None]
            faces = cube_mesh.faces_packed()[None]

        return verts, faces

    @staticmethod
    def generate_unit_cube_mesh_from_file(file_name, device):
        # .obj file of cylinder is located in the same directory as this file in primitives/cylinder.obj
        current_directory = os.path.dirname(os.path.realpath(__file__))
        cube_obj_path = os.path.join(current_directory, 'primitives/' + file_name)

        with open(cube_obj_path, 'r') as f:
            verts, faces, _ = load_obj(f, load_textures=False, device=device)

        return verts[None], faces.verts_idx[None]


class NodeMeshPrimitiveCube(Node):
    def __init__(self, primitive_id, bpy_node, config):
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
        super().__init__(bpy_node, config)
        print('Creating MeshPrimitiveCube')

        # self.default_cube = PrimitiveCube(primitive_id, device=self.device)
        self.default_cube = PrimitiveCube(primitive_id, primitive_filename='cube_8.obj', device=self.device)

        self.cached_output = None

    def to(self, device):
        super().to(device)

        self.default_cube.to(device)

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
                                      NodeMeshPrimitiveCubeStrings.Size_str]
        cube_primitive = self.default_cube.clone()

        assert cube_primitive.get_device() == size.device
        assert size.shape[0] == 1, "Batch size > 1 is not supported yet."

        cube_primitive.scale(size)

        inputs_dict[self.name][NodeStrings.OUT_str + 'Mesh'] = [cube_primitive]

        if not len(self.in_edges):
            self.cached_output = {NodeStrings.OUT_str + 'Mesh': [cube_primitive]}

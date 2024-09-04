from pytorch3d.structures import Meshes

from PytorchGeoNodes.Nodes.Node import *
from PytorchGeoNodes.Nodes.node_types import *

class NodeGroupOutput(Node):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Group Output Node adds output to the dict of previously added outputs.

        :param bpy_node:
        """
        super().__init__(bpy_node)

        print('Creating NodeGroupOutput')

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        output_meshes = []
        for output_key in inputs_dict[self.name].keys():
            b_meshes = []
            for mesh in inputs_dict[self.name][output_key]:
                assert isinstance(mesh, Meshes), 'Only Meshes are supported as outputs for now'

                b_meshes.append(mesh)

            assert output_key[len(NodeStrings.IN_str):] \
                   not in inputs_dict[NodeTypes.Geometry_Nodes_Outputs_str].keys(), \
                f'Output {output_key[len(NodeStrings.IN_str):]} already exists in inputs_dict. '
            inputs_dict[NodeTypes.Geometry_Nodes_Outputs_str][output_key[len(NodeStrings.IN_str):]] = b_meshes

            output_meshes.append(b_meshes)

        return output_meshes

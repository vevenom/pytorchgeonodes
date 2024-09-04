import bpy
import torch

from PytorchGeoNodes.Nodes.Node import *
from PytorchGeoNodes.Nodes.node_types import NodeTypes

class NodeGroup(Node):
    def __init__(self, bpy_node: bpy.types.GeometryNode, geometry_nodes):
        """
        Node that groups other nodes.

        :param bpy_node:
        """
        super().__init__(bpy_node)

        print('Creating NodeGroup')

        self.geometry_nodes = geometry_nodes

    def to(self, device):
        super().to(device)
        self.geometry_nodes.to(device)

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        new_geo_nodes_inputs_dict = {
        }
        for edge in self.in_edges:

            from_socket = edge.from_socket
            to_socket = edge.to_socket

            new_geo_nodes_inputs_dict[to_socket.name] = \
                new_geo_nodes_inputs_dict[to_socket.identifier] = (
                inputs_dict)[self.name][NodeStrings.IN_str + to_socket.identifier]

        geo_nodes_inputs_dict, _ = self.geometry_nodes.forward(new_geo_nodes_inputs_dict)

        for edge in self.out_edges:

            from_socket = edge.from_socket
            # to_socket = edge.to_socket
            # from_node = edge.from_node

            # print(edge.from_node, from_socket.name)
            inputs_dict[self.name][NodeStrings.OUT_str + from_socket.identifier] = (
                geo_nodes_inputs_dict)[NodeTypes.Geometry_Nodes_Outputs_str][from_socket.identifier]



import bpy
from torch import nn
import torch

from PytorchGeoNodes.Nodes.node_types import NodeTypes

class NodeStrings:
    VALUE_type_str = 'VALUE'
    VECTOR_type_str = 'VECTOR'
    GEOMETRY_type_str = 'GEOMETRY'
    FLOAT_type_str = 'FLOAT'
    RGBA_type_str = 'RGBA'
    STRING_type_str = 'STRING'
    OBJECT_type_str = 'OBJECT'
    CUSTOM_type_str = 'CUSTOM'
    COLLECTION_type_str = 'COLLECTION'
    TEXTURE_type_str = 'TEXTURE'
    MATERIAL_type_str = 'MATERIAL'
    IMAGE_type_str = 'IMAGE'
    ROTATION_type_str = 'ROTATION'
    INT_type_str = 'INT'
    BOOLEAN_type_str = 'BOOLEAN'
    IN_str = 'In'
    OUT_str = 'Out'

class Node(nn.Module):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        super().__init__()

        self.device = 'cpu'

        self.type = bpy_node.type
        self.name = bpy_node.name

        self.in_edges = []
        self.out_edges = []

        self.input_params_dict = {}
        for input in bpy_node.inputs:
            if input.type in [NodeStrings.VALUE_type_str,
                              NodeStrings.VECTOR_type_str,
                              NodeStrings.INT_type_str,
                              NodeStrings.BOOLEAN_type_str]:
                self.input_params_dict[input.identifier] = torch.tensor([[input.default_value]])
            elif input.type == NodeStrings.GEOMETRY_type_str:
                self.input_params_dict[input.identifier] = None
            elif input.type == NodeStrings.CUSTOM_type_str:
                # Not sure what CUSTOM is for yet and
                # seems to do nothing
                continue
            elif input.type == NodeStrings.RGBA_type_str:
                self.input_params_dict[input.identifier] = input.default_value
                print("[WARNING] RGBA input type not supported yet this is just a placeholder")
            elif input.type == NodeStrings.STRING_type_str:
                self.input_params_dict[input.identifier] = input.default_value
                print("[WARNING] STRING input type not supported yet this is just a placeholder")
            elif input.type == NodeStrings.OBJECT_type_str:
                self.input_params_dict[input.identifier] = input.default_value
                print("[WARNING] OBJECT input type not supported yet this is just a placeholder")
            elif input.type == NodeStrings.COLLECTION_type_str:
                self.input_params_dict[input.identifier] = input.default_value
                print("[WARNING] COLLECTION input type not supported yet this is just a placeholder")
            elif input.type == NodeStrings.TEXTURE_type_str:
                self.input_params_dict[input.identifier] = input.default_value
                print("[WARNING] TEXTURE input type not supported yet this is just a placeholder")
            elif input.type == NodeStrings.MATERIAL_type_str:
                self.input_params_dict[input.identifier] = input.default_value
                print("[WARNING] MATERIAL input type not supported yet this is just a placeholder")
            elif input.type == NodeStrings.IMAGE_type_str:
                self.input_params_dict[input.identifier] = input.default_value
                print("[WARNING] IMAGE input type not supported yet this is just a placeholder")
            elif input.type == NodeStrings.ROTATION_type_str:
                self.input_params_dict[input.identifier] = input.default_value
                print("[WARNING] ROTATION input type not supported yet this is just a placeholder")
            else:
                # print(input.identifier)
                raise Exception(f'Input type {input.type} is not supported yet.')

        self.use_cache = False

        # print('self Node', dill.pickles(self))

    def __str__(self):
        return f'Node: {self.name}, type: {self.type}'

    def assign_cachable(self):
        if self.type == NodeTypes.GROUP_INPUT_str:
            self.use_cache = False
        elif not len(self.in_edges):
            self.use_cache = True
        else:
            all_inputs_have_cache = True
            for edge in self.in_edges:
                    if not edge.from_node.use_cache:
                        all_inputs_have_cache = False
                        break
            self.use_cache = all_inputs_have_cache

    def to(self, device):
        super().to(device)
        self.device = device
        for key in self.input_params_dict.keys():
            if self.input_params_dict[key] is not None and hasattr(self.input_params_dict[key], 'device'):
                self.input_params_dict[key] = self.input_params_dict[key].to(device)

    def forward(self, inputs_dict):
        inputs_dict[self.name] = {}

        for in_edge in self.in_edges:

            from_node = in_edge.from_node
            # to_node = in_edge.to_node
            from_socket = in_edge.from_socket
            to_socket = in_edge.to_socket

            if to_socket.identifier in self.input_params_dict.keys():
                assert not NodeStrings.IN_str + to_socket.identifier in inputs_dict[self.name].keys(), \
                    f'Input {to_socket.identifier} already assigned to {from_node.name}.'
                if NodeStrings.IN_str + to_socket.identifier in inputs_dict[self.name].keys():
                    assert isinstance(inputs_dict[self.name][NodeStrings.IN_str + to_socket.identifier], list)
                    assert isinstance(inputs_dict[from_node.name][NodeStrings.OUT_str + from_socket.identifier], list)
                    inputs_dict[self.name][NodeStrings.IN_str + to_socket.identifier] += \
                        inputs_dict[from_node.name][NodeStrings.OUT_str + from_socket.identifier]
                else:
                    inputs_dict[self.name][NodeStrings.IN_str + to_socket.identifier] = \
                        inputs_dict[from_node.name][NodeStrings.OUT_str + from_socket.identifier]

        for key in self.input_params_dict.keys():
            if NodeStrings.IN_str + key not in inputs_dict[self.name].keys():
                inputs_dict[self.name][NodeStrings.IN_str + key] = self.input_params_dict[key]

    def add_in_edge(self, edge):
        self.in_edges.append(edge)

    def add_out_edge(self, edge):
        self.out_edges.append(edge)

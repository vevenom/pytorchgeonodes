import bpy
import torch

from PytorchGeoNodes.Nodes.Node import *
from PytorchGeoNodes.Nodes.node_types import NodeTypes
from PytorchGeoNodes.BlenderShapeProgram import BlenderShapeProgram

class NodeGroupInput(Node):
    def __init__(self, bpy_node: bpy.types.GeometryNode, params_dict):
        """
        The Group Input Node groups inputs .

        :param bpy_node:
        """
        super().__init__(bpy_node)

        print('Creating NodeGroupInput')

        self.input_params_dict = {}
        self.identifier_to_name = {}
        for param in bpy_node.outputs:
            if param.type == 'CUSTOM':
                # Not sure what CUSTOM is for yet, it corresponds to NodeSocketVirtual and
                # seems to do nothing
                continue
            # self.params.append(torch.tensor(param.default_value))
            if param.name in params_dict.keys():
                self.input_params_dict[param.identifier] = torch.tensor(params_dict[param.name])
            else:
                self.input_params_dict[param.identifier] = None
            self.identifier_to_name[param.identifier] = param.name

    def forward(self, inputs_dict):
        inputs_dict[self.name] = {}

        for key in self.input_params_dict.keys():
            if key not in inputs_dict[NodeTypes.Geometry_Nodes_Inputs_str].keys():
                inputs_dict[self.name][NodeStrings.OUT_str + key] = self.input_params_dict[key][None, None].to(self.device)
            else:
                inputs_dict[self.name][NodeStrings.OUT_str + key] = \
                    inputs_dict[NodeTypes.Geometry_Nodes_Inputs_str][key]


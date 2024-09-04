import bpy
import torch

from PytorchGeoNodes.Nodes.Node import *


class NodeCombXYZ(Node):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Combine XYZ Node combines a vector from its individual components.

        Inputs: X, Y, Z
        Outputs: Vector

        :param bpy_node:
        """
        super().__init__(bpy_node)
        print('Creating NodeCombXYZ')

        self.cached_output = None

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        if self.cached_output is not None:
            inputs_dict[self.name][NodeStrings.OUT_str + 'Vector'] = \
                self.cached_output[NodeStrings.OUT_str + 'Vector']
            return

        xyz = torch.zeros((inputs_dict[self.name][NodeStrings.IN_str + 'X'].shape[0], 1, 3),
                          device=inputs_dict[self.name][NodeStrings.IN_str + 'X'].device)
        xyz[:, 0, 0] = inputs_dict[self.name][NodeStrings.IN_str + 'X'][:, 0]
        xyz[:, 0, 1] = inputs_dict[self.name][NodeStrings.IN_str + 'Y'][:, 0]
        xyz[:, 0, 2] = inputs_dict[self.name][NodeStrings.IN_str + 'Z'][:, 0]

        inputs_dict[self.name][NodeStrings.OUT_str + 'Vector'] = xyz

        if self.cached_output is not None and (not len(self.in_edges)):
            self.cached_output = {NodeStrings.OUT_str + 'Vector': xyz}

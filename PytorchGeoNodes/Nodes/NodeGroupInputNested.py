from PytorchGeoNodes.Nodes.Node import *
from PytorchGeoNodes.Nodes.node_types import *

class NodeGroupInputNested(Node):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Group Input Node groups inputs .

        :param bpy_node:
        """
        super().__init__(bpy_node)

        print('Creating NodeGroupInputNested')

        self.input_params_dict = {}
        for param in bpy_node.outputs:
            if param.type == 'CUSTOM':
                # Not sure what CUSTOM is for yet, it corresponds to NodeSocketVirtual and
                # seems to do nothing
                continue
            self.input_params_dict[param.identifier] = None

    def forward(self, inputs_dict):
        inputs_dict[self.name] = {}
        for key in self.input_params_dict.keys():
            assert key in inputs_dict[NodeTypes.Geometry_Nodes_Inputs_str].keys(), \
                (f'Input {key} not found in inputs_dict. ' +
                 f'The input is not assigned any value. Available inputs are: ' +
                 f'{list(inputs_dict[NodeTypes.Geometry_Nodes_Inputs_str].keys())}')

            inputs_dict[self.name][NodeStrings.OUT_str + key] = \
                inputs_dict[NodeTypes.Geometry_Nodes_Inputs_str][key]


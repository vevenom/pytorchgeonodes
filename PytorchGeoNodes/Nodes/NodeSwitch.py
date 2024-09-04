from pytorch3d.structures import Meshes

from PytorchGeoNodes.Nodes.Node import *


class NodeSwitchStrings:
    Switch_str = 'Switch'
    Switch_001_str = 'Switch_001'

    # Geometry switch strings
    False_006_str = 'False_006'
    True_006_str = 'True_006'


class NodeSwitch(Node):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Switch node outputs one of two inputs depending on a condition.
        Only the input that is passed through the node is computed.

        :param bpy_node:
        """
        super().__init__(bpy_node)

        self.switch_type = bpy_node.input_type

        self.allowed_input_types = allowed_input_types = [
            NodeStrings.GEOMETRY_type_str,
            NodeStrings.FLOAT_type_str,
            NodeStrings.VECTOR_type_str]
        if bpy_node.input_type not in allowed_input_types:
            # TODO: when implementing new node input types, remember to change
            #  the base class constructor to do proper parsing
            raise NotImplementedError(f'NodeSwitch only implemented for {allowed_input_types} types. ' +
                                      f'Input type is {bpy_node.input_type}.')

        # TODO some switch types are unhashable. Delete them from input_params_dict for now:
        # TODO In case these are needed in future they need to be converted to a pickable type
        # False_012, True_012, False_004, True_004
        del self.input_params_dict['False_004']
        del self.input_params_dict['True_004']
        del self.input_params_dict['False_012']
        del self.input_params_dict['True_012']

        print('Creating NodeSwitch')

        self.cached_output = None

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        if self.switch_type == NodeStrings.GEOMETRY_type_str:
            switch_bool = inputs_dict[self.name][NodeStrings.IN_str +
                                                 NodeSwitchStrings.Switch_001_str][:, 0]

            true_values = inputs_dict[self.name][NodeStrings.IN_str +
                                                 NodeSwitchStrings.True_006_str]
            false_values = inputs_dict[self.name][NodeStrings.IN_str +
                                                  NodeSwitchStrings.False_006_str]

            meshes = []
            for i in range(switch_bool.shape[0]):
                if switch_bool[i]:
                    if true_values is None:
                        meshes.append(Meshes(verts=[], faces=[]).to(switch_bool.device))
                    else:
                        meshes.append(true_values[i])
                else:
                    if false_values is None:
                        meshes.append(Meshes(verts=[], faces=[]).to(switch_bool.device))
                    else:
                        meshes.append(false_values[i])

            # TODO: Is it always called Output_006?
            inputs_dict[self.name][NodeStrings.OUT_str + 'Output_006'] = meshes
        elif self.switch_type == NodeStrings.FLOAT_type_str:
            switch_bool = inputs_dict[self.name][NodeStrings.IN_str +
                                                 NodeSwitchStrings.Switch_str]


            # print(inputs_dict[self.name])
            # print(inputs_dict[self.name].keys())
            # assert False

            true_values = inputs_dict[self.name][NodeStrings.IN_str + 'True']
            false_values = inputs_dict[self.name][NodeStrings.IN_str + 'False']

            inputs_dict[self.name][NodeStrings.OUT_str + 'Output'] = torch.zeros_like(false_values)
            inputs_dict[self.name][NodeStrings.OUT_str + 'Output'][switch_bool] = true_values[switch_bool]
            inputs_dict[self.name][NodeStrings.OUT_str + 'Output'][torch.logical_not(switch_bool)] = (
                false_values)[torch.logical_not(switch_bool)]

        elif self.switch_type == NodeStrings.VECTOR_type_str:
            switch_bool = inputs_dict[self.name][NodeStrings.IN_str +
                                                 NodeSwitchStrings.Switch_str][:, 0]

            true_values = inputs_dict[self.name][NodeStrings.IN_str + 'True_003']
            false_values = inputs_dict[self.name][NodeStrings.IN_str + 'False_003']

            inputs_dict[self.name][NodeStrings.OUT_str + 'Output_003'] = false_values
            inputs_dict[self.name][NodeStrings.OUT_str + 'Output_003'][switch_bool] = true_values[switch_bool]
        else:
            # TODO: when implementing new node input types, remember to implement behavior here
            raise NotImplementedError(f'NodeSwitch only implemented for {self.allowed_input_types} type. ' +
                                      f'Input type is {self.switch_type}.')
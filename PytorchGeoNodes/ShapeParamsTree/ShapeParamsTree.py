import graphviz
import numpy as np
from abc import abstractmethod

class ShapeParamNode(object):
    def __init__(self, name, node_type, children=None):
        if children is None:
            children = []

        self.name = name
        self.type = node_type
        self.current_value = None
        self.children = children
        self.value = None

    def append_child(self, child):
        self.children.append(child)

    def init_node_from_config(self, config):
        """
        Builds a tree from a config file
        :param config: json config
        :return:
        """
        # Generate a node from config dict. For every key in config create a child node:
        #  - name: key
        #  - type: self.config['params'][key]['type']
        #  - if type is 'nested', then generate children nodes
        #  - if type is 'float', 'int' or 'bool', then generate a leaf node and set valid_values.

        params_config = config['params']
        for param_name in params_config.keys():
            or_dependencies = params_config[param_name]['or_dependencies'] \
                if 'or_dependencies' in params_config[param_name].keys() else None
            not_dependencies = params_config[param_name]['not_dependencies'] \
                if 'not_dependencies' in params_config[param_name].keys() else None

            if params_config[param_name]['type'] == 'nested':
                child = ShapeParamNode(param_name, 'nested')
                child.init_node_from_config(params_config[param_name])
                self.append_child(child)
            elif params_config[param_name]['type'] == 'float':
                child = FloatParamNode(param_name,
                                       valid_range=params_config[param_name]['range'],
                                       or_dependencies=or_dependencies,
                                       not_dependencies=not_dependencies)
                self.append_child(child)
            elif params_config[param_name]['type'] == 'int':
                if 'range' in params_config[param_name].keys():
                    params_valid_values = params_config[param_name]['range']
                    params_valid_values = set(range(params_valid_values[0], params_valid_values[1]))
                elif 'valid_values' in params_config[param_name].keys():
                    params_valid_values = set(params_config[param_name]['valid_values'])
                else:
                    raise ValueError("Invalid config for parameter %s with value %s" %
                                     (param_name, params_config[param_name]))

                child = IntParamNode(param_name, valid_values=params_valid_values,
                                     or_dependencies=or_dependencies,
                                     not_dependencies=not_dependencies)
                self.append_child(child)
            elif params_config[param_name]['type'] == 'bool':
                child = BoolParamNode(param_name, or_dependencies=or_dependencies,
                                      not_dependencies=not_dependencies)
                self.append_child(child)
            else:
                raise NotImplementedError("Parameter type %s not implemented" % params_config[param_name]['type'])

    def _visualize_node_graphviz_(self, dot: graphviz.Digraph, show_values=True):
        """
        Visualizes the node using graphviz library

        :param dot: graphviz object
        :param show_values: if True, then show values of the nodes
        :return:
        """

        # 1) draw node
        # 2) draw its children
        dot.node(self.name, self.name)
        if show_values and self.type != 'nested':
            if self.value is not None:
                if self.type == np.float_:
                    value_str = "{0:.3f}".format(self.value)
                else:
                    value_str = str(self.value)
                dot.node(self.name + '_value', value_str, shape='rectangle', color='blue')
            else:
                dot.node(self.name + '_value', 'None', shape='rectangle', color='red')
            dot.edge(self.name, self.name + '_value', arrowhead='none')

        for child in self.children:
            child._visualize_node_graphviz_(dot, show_values=show_values)
            dot.edge(self.name, child.name)

    @abstractmethod
    def set_value(self, value):
        """
        Sets the value of the node and checks if it is valid

        :param value: Value to set
        :return:
        """
        raise NotImplementedError("set_value not implemented")

    def get_value(self):
        """
        Get the value and check if it is set

        :return: return node value
        """

        # check if value is set
        if self.value is None:
            raise ValueError("Value is not set")
        return self.value

    @abstractmethod
    def get_normalized_value(self):
        """
        Get the value and normalize it

        :return: Normalized value
        """

    def set_value_from_dict(self, params_dict, is_normalized, set_children_below=False):
        """
        Sets the value of the node from a dictionary

        :param params_dict:
        :param is_normalized:
        :param set_children_below:
        :return:
        """

        if self.type == 'nested':
            if set_children_below:
                for child in self.children:
                    child.set_value_from_dict(params_dict, is_normalized, set_children_below=set_children_below)
        else:
            if self.name not in params_dict.keys():
                raise ValueError("Parameter %s not found in params_dict" % self.name)

            if is_normalized:
                self.set_value(params_dict[self.name], is_normalized=is_normalized)
            else:
                self.set_value(params_dict[self.name], is_normalized=False)

    @abstractmethod
    def get_node_meta(self):
        """
        Get the node meta information

        :return:
        """
        raise NotImplementedError("get_node_meta is an abstract method")

    def reset_node(self, reset_all_nodes_below=False):
        self.value = None
        if reset_all_nodes_below:
            for child in self.children:
                child.reset_node(reset_all_nodes_below=reset_all_nodes_below)

    def randomize_node_value(self, randomize_all_nodes_below=False):
        """
        Randomizes node values

        :param randomize_all_nodes_below:
        :return:
        """

        if randomize_all_nodes_below:
            for child in self.children:
                child.randomize_node_value(randomize_all_nodes_below=randomize_all_nodes_below)

    def get_all_descendants(self, leaf_only=True):
        """
        Get all descendants of the node

        :param leaf_only:
        :return:
        """

        if leaf_only and self.type != 'nested':
            return [self]
        else:
            descendants = []
            for child in self.children:
                descendants += child.get_all_descendants(leaf_only=leaf_only)
            return descendants

    def to_params_dict_(self, leaf_only=True, get_normalized_values=False):
        """
        Convert the node to a dictionary of parameters

        :param leaf_only:
        :return:
        """

        if leaf_only and self.type != 'nested':
            if get_normalized_values:
                return {self.name: self.get_normalized_value()}
            else:
                return {self.name: self.get_value()}
        else:
            params_dict = {}
            for child in self.children:
                params_dict.update(child.to_params_dict_(leaf_only=leaf_only,
                                                         get_normalized_values=get_normalized_values))
            return params_dict


class FloatParamNode(ShapeParamNode):
    def __init__(self, name, valid_range, or_dependencies=None, not_dependencies=None, children=None):
        super().__init__(name, float, children)

        self.or_dependencies = or_dependencies
        self.not_dependencies = not_dependencies

        self.valid_range = valid_range

    def get_normalized_value(self):
        """
        Get the value and normalize it to be in range [-1,1]
        :return:
        """
        value = self.get_value()
        return (value - self.valid_range[0]) / (self.valid_range[1] - self.valid_range[0]) * 2 - 1

    def get_node_meta(self):
        """
        Get the node meta information

        :return:
        """
        return {'valid_range': self.valid_range,
                'or_dependencies': self.or_dependencies,
                'not_dependencies': self.not_dependencies,
                'type': self.type}

    def set_value(self, value, is_normalized=False):
        assert np.isscalar(value) or value.size == 1, "Value %s is not a scalar" % str(value)

        if is_normalized:
            assert value >= -1 and value <= 1, "Value %f is not in range [-1, 1]" % value
            value = (value + 1) / 2 * (self.valid_range[1] - self.valid_range[0]) + self.valid_range[0]

        if value < self.valid_range[0] or value > self.valid_range[1]:
            raise ValueError("Value %f is not in range [%f, %f]" %
                             (value, self.valid_range[0], self.valid_range[1]))

        if not np.isscalar(value):
            value = value.reshape(-1)[0]
        self.value = self.type(value)

    def randomize_node_value(self, randomize_all_nodes_below=False):
        """
        Randomizes node value to be in range [self.valid_range[0], self.valid_range[1]]

        :param randomize_all_nodes_below:
        :return:
        """

        value = np.random.uniform(low=self.valid_range[0], high=self.valid_range[1])
        self.set_value(value)

        super().randomize_node_value(randomize_all_nodes_below=randomize_all_nodes_below)


class IntParamNode(ShapeParamNode):
    def __init__(self, name, valid_values, or_dependencies=None, not_dependencies=None, children=None):
        super().__init__(name, int, children)

        self.valid_values = valid_values
        self.not_dependencies = not_dependencies

        self.or_dependencies = or_dependencies

        self.value_to_ind = {value: ind for ind, value in enumerate(valid_values)}

    def get_num_valid_value(self):
        return len(self.valid_values)

    def get_normalized_value(self):
        """
        Get the value and normalize it to be in range [0, len(valid_values) - 1]

        :return:
        """
        return self.value_to_ind[self.get_value()]

    def get_node_meta(self):
        """
        Get the node meta information

        :return:
        """
        return {'valid_values': self.valid_values,
                'or_dependencies': self.or_dependencies,
                'not_dependencies': self.not_dependencies,
                'type': self.type}

    def set_value(self, value, is_normalized=False):
        assert np.isscalar(value) or value.size == 1, "Value %s is not a scalar" % str(value)

        if not np.isscalar(value):
            value = value.reshape(-1)[0]

        if is_normalized:
            value = np.round(value)
            value = list(self.valid_values)[int(value)]

        if value not in self.valid_values:
            raise ValueError("Value %d is not in valid values %s for %s" % (value, str(self.valid_values), self.name))
        self.value = self.type(value)

    def randomize_node_value(self, randomize_all_nodes_below=False):
        """
        Randomizes node values

        :param randomize_all_nodes_below:
        :return:
        """

        if isinstance(self.valid_values, list):
            # self.value = np.random.randint(low=self.valid_values[0], high=self.valid_values[1])
            # print(self.valid_values)
            raise NotImplementedError("randomize_node_value() not implemented for list")
        else:

            value = self.type(np.random.choice(list(self.valid_values)))
            self.set_value(value)
        super().randomize_node_value(randomize_all_nodes_below=randomize_all_nodes_below)


class BoolParamNode(ShapeParamNode):
    def __init__(self, name, or_dependencies=None, not_dependencies=None, children=None):
        super().__init__(name, bool, children)

        self.or_dependencies = or_dependencies
        self.not_dependencies = not_dependencies

    def get_normalized_value(self):
        """
        Get the value and "normalize" (convert) it to 0 or 1

        :return:
        """
        return int(self.get_value())

    def get_node_meta(self):
        """
        Get the node meta information

        :return:
        """
        return {
            'or_dependencies': self.or_dependencies,
            'not_dependencies': self.not_dependencies,
            'type': self.type}

    def set_value(self, value, is_normalized=False):
        if is_normalized:
            value = np.round(value)
            value = bool(value)

        if value not in [False, True]:
            raise ValueError("Value %d is not in [False, True]" % value)
        value = self.type(value)
        self.value = value

    def randomize_node_value(self, randomize_all_nodes_below=False):
        """
        Randomizes node values

        :param randomize_all_nodes_below:
        :return:
        """

        value = np.random.choice([False, True])
        self.set_value(value)

        super().randomize_node_value(randomize_all_nodes_below=randomize_all_nodes_below)

class ShapeParamsTree(object):
    def __init__(self):
        self.root = ShapeParamNode('root', 'nested')
        self.name = '' # name of the tree

    def build_tree_from_config(self, config):
        """
        Builds a tree from a config file

        :param config: json config
        :return:
        """
        self.root.init_node_from_config(config)
        self.name = config['class_name']

    def visualize_tree(self, show_values=True):
        """
        Visualizes the tree
        :return:
        """
        dot = graphviz.Digraph(comment='Shape Params Tree')
        self.root._visualize_node_graphviz_(dot, show_values=show_values)
        dot.render('/tmp/shape_params_tree.gv', view=True)

    def reset_tree(self):
        """
        Resets tree values

        :return:
        """
        self.root.reset_node(reset_all_nodes_below=True)

    def randomize_tree_values(self, new_seed=None):
        """
        Randomizes tree values

        :return:
        """

        if new_seed is not None:
            np.random.seed(new_seed)

        self.root.randomize_node_value(randomize_all_nodes_below=True)

    def get_leaf_nodes(self):
        """
        Get all leaf nodes

        :return: list of leaf nodes
        :rtype: list[ShapeParamNode]
        """

        return self.root.get_all_descendants(leaf_only=True)

    def get_nodes_meta(self):
        """
        Get the node meta information

        :return:
        """
        nodes = self.root.get_all_descendants(leaf_only=True)
        nodes_meta = {}
        for node in nodes:
            nodes_meta[node.name] = node.get_node_meta()
        return nodes_meta

    def to_params_dict(self, get_normalized_values=False):
        return self.root.to_params_dict_(leaf_only=True, get_normalized_values=get_normalized_values)

    def set_values_from_dict(self, params_dict, is_normalized):
        """
        Sets values from a dict

        :param params_dict: dict of params
        :param is_normalized: whether the values are normalized
        :return:
        """

        self.root.set_value_from_dict(params_dict, is_normalized=is_normalized, set_children_below=True)


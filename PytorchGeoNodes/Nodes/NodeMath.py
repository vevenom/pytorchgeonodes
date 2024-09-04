from PytorchGeoNodes.Nodes.Node import *

class NodeMathStrings:
    ADD_str = 'ADD'
    SUBTRACT_str = 'SUBTRACT'
    MULTIPLY_str = 'MULTIPLY'
    DIVIDE_str = 'DIVIDE'
    COMPARE_STR = 'COMPARE'
    VALUE_str = 'Value'
    VALUE_001_str = 'Value_001'
    VALUE_002_str = 'Value_002'


class NodeOperatorAdd(nn.Module):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Add Operator adds two numbers.

        :param bpy_node:
        """
        super().__init__()

        print('Creating NodeAdd operator')

    def forward(self, inputs_dict):
        x = inputs_dict[NodeStrings.IN_str + NodeMathStrings.VALUE_str]
        y = inputs_dict[NodeStrings.IN_str + NodeMathStrings.VALUE_001_str]

        res = x + y
        return res

class NodeOperatorSubtract(nn.Module):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Subtract Operator subtracts two numbers.

        :param bpy_node:
        """
        super().__init__()

        print('Creating NodeSubtract operator')

    def forward(self, inputs_dict):
        x = inputs_dict[NodeStrings.IN_str + NodeMathStrings.VALUE_str]
        y = inputs_dict[NodeStrings.IN_str + NodeMathStrings.VALUE_001_str]

        res = x - y
        return res

class NodeOperatorMultiply(nn.Module):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Multiply Operator multiplies two numbers.

        :param bpy_node:
        """
        super().__init__()

        print('Creating NodeMultiply operator')

    def forward(self, inputs_dict):
        x = inputs_dict[NodeStrings.IN_str + NodeMathStrings.VALUE_str]
        y = inputs_dict[NodeStrings.IN_str + NodeMathStrings.VALUE_001_str]

        res = x * y
        return res


class NodeOperatorDivide(nn.Module):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Divide Operator divides two numbers.



        :param bpy_node:
        """
        super().__init__()

        print('Creating NodeDivide operator')

    def forward(self, inputs_dict):
        x = inputs_dict[NodeStrings.IN_str + NodeMathStrings.VALUE_str]
        y = inputs_dict[NodeStrings.IN_str + NodeMathStrings.VALUE_001_str]

        eps = 1e-4
        assert torch.all(torch.abs(y) > eps), f'abs(y) must be greater than {eps}'
        res = x / y
        return res


class NodeOperatorCompare(nn.Module):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Compare Operator compares two numbers.

        :param bpy_node:
        """

        super().__init__()

        print('Creating NodeCompare operator')

        self.eps = 1e-4  # Epsilon for comparison

    def forward(self, inputs_dict):

        x = inputs_dict[NodeStrings.IN_str + NodeMathStrings.VALUE_str]
        y = inputs_dict[NodeStrings.IN_str + NodeMathStrings.VALUE_001_str]

        device = x.device
        res = torch.where(torch.abs(x - y) < self.eps,
                          torch.tensor(True, device=device),
                          torch.tensor(False, device=device))

        return res


class NodeMath(Node):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Combine XYZ Node combines a vector from its individual components.

        Inputs: X, Y, Z
        Outputs: Vector

        :param bpy_node:
        """
        super().__init__(bpy_node)

        print('Creating NodeMath')

        if bpy_node.operation == NodeMathStrings.ADD_str:
            self.op = NodeOperatorAdd(bpy_node)
        elif bpy_node.operation == NodeMathStrings.SUBTRACT_str:
            self.op = NodeOperatorSubtract(bpy_node)
        elif bpy_node.operation == NodeMathStrings.MULTIPLY_str:
            self.op = NodeOperatorMultiply(bpy_node)
        elif bpy_node.operation == NodeMathStrings.DIVIDE_str:
            self.op = NodeOperatorDivide(bpy_node)
        elif bpy_node.operation == NodeMathStrings.COMPARE_STR:
            self.op = NodeOperatorCompare(bpy_node)
        else:
            raise Exception(f'Operation {bpy_node.operation} is not supported yet.')

        self.cached_output = None

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        if not len(self.in_edges) and self.cached_output is not None:
            inputs_dict[self.name][NodeStrings.OUT_str + 'Value'] = \
                self.cached_output[NodeStrings.OUT_str + 'Value']
            return

        res = self.op(inputs_dict[self.name])

        inputs_dict[self.name][NodeStrings.OUT_str + 'Value'] = res

        if not len(self.in_edges):
            self.cached_output = {NodeStrings.OUT_str + 'Value': res}

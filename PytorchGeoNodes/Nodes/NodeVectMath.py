from PytorchGeoNodes.Nodes.Node import *

class NodeVectMathStrings:
    DIVIDE_str = 'DIVIDE'
    VECTOR_str = 'Vector'
    VECTOR_001_str = 'Vector_001'
    Vector_002_str = 'Vector_002'


class NodeOperatorDivide(nn.Module):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Divide Operator divides two numbers.



        :param bpy_node:
        """
        super().__init__()

        print('Creating NodeDivide operator')

    def forward(self, inputs_dict):
        x = inputs_dict[NodeStrings.IN_str +
                        NodeVectMathStrings.VECTOR_str]
        y = inputs_dict[NodeStrings.IN_str +
                        NodeVectMathStrings.VECTOR_001_str]

        eps = 1e-4
        assert torch.all(torch.abs(y) > eps), f'abs(y) must be greater than {eps}'
        res = x / y
        return res

class NodeVectMath(Node):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Vector Math node performs the selected math operation on the input vectors.

        :param bpy_node:
        """
        super().__init__(bpy_node)

        print('Creating NodeVectMath')

        if bpy_node.operation == NodeVectMathStrings.DIVIDE_str:
            self.op = NodeOperatorDivide(bpy_node)
        else:
            raise Exception(f'Operation {bpy_node.operation} is not supported yet.')

        self.cached_output = None

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        if not len(self.in_edges) and self.cached_output is not None:
            inputs_dict[self.name][NodeStrings.OUT_str + 'Vector'] = \
                self.cached_output[NodeStrings.OUT_str + 'Vector']

        res = self.op(inputs_dict[self.name])

        inputs_dict[self.name][NodeStrings.OUT_str + 'Vector'] = res

        if not len(self.in_edges):
            self.cached_output = {NodeStrings.OUT_str +
                                  'Vector': inputs_dict[self.name][NodeStrings.OUT_str + 'Vector']}
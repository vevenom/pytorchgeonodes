from PytorchGeoNodes.Nodes.Node import *

class NodeMeshPrimitiveLineStrings:
    OFFSET_str = 'OFFSET'

class NodeMeshPrimitiveLine(Node):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Mesh Line node generates vertices in a line and connects them with edges.

        Inputs:
            -- Counts: Number of vertices on the line.
            -- Start Location: Position of the first vertex. (X, Y, Z)
            -- Offset: Offset of the vertices along the line. (X, Y, Z)

        Outputs: Vector

        :param bpy_node:
        """
        super().__init__(bpy_node)

        print('Creating MeshPrimitiveLine')

        self.cached_output = None

        assert bpy_node.mode == \
               NodeMeshPrimitiveLineStrings.OFFSET_str, f'Mode {bpy_node.mode} is not supported yet.'

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        if not len(self.in_edges) and self.cached_output is not None:
            inputs_dict[self.name][NodeStrings.OUT_str + 'Mesh'] = \
                self.cached_output[NodeStrings.OUT_str + 'Mesh']
            return

        start_location = inputs_dict[self.name][NodeStrings.IN_str + 'Start Location']
        count = inputs_dict[self.name][NodeStrings.IN_str + 'Count']
        offset = inputs_dict[self.name][NodeStrings.IN_str + 'Offset']

        line_vertices = []
        for b in range(count.shape[0]):
            line_vertices_b = \
                torch.add(start_location[b],
                          torch.mul(offset[b],
                                    torch.arange(count[b, 0], device=self.device).unsqueeze(1)))

            line_vertices.append(line_vertices_b)

        inputs_dict[self.name][NodeStrings.OUT_str + 'Mesh'] = line_vertices

        if not len(self.in_edges):
            self.cached_output = {NodeStrings.OUT_str + 'Mesh': line_vertices}


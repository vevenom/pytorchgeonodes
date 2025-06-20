from PytorchGeoNodes.Nodes.Node import *
from ProceduralGaussians.GaussianNodes.PrimitiveGaussian import PrimitiveGaussian
from PytorchGeoNodes.Nodes.NodeMeshPrimitiveCylinder import PrimitiveCylinder

class NodeMeshPrimitiveCylinderStrings:
    Vertices_str = 'Vertices'
    Side_Segments_str = 'Side Segments'
    Fill_Segments_str = 'Fill Segments'
    Radius_str = 'Radius'
    Depth_str = 'Depth'


class NodeGaussianPrimitiveCylinder(Node):
    def __init__(self, identifier, bpy_node, config):
        """
        The Cylinder node generates a cylinder mesh. It is similar to the Cone node but always uses the same radius
        for the circles at the top and bottom.

        Inputs:
            -- Counts: Number of vertices on the line.
            -- Start Location: Position of the first vertex. (X, Y, Z)
            -- Offset: Offset of the vertices along the line. (X, Y, Z)

        Outputs: Vector

        :param bpy_node:
        """
        super().__init__(bpy_node, config)
        print('Creating MeshPrimitiveCylinder')

        primitive_cylinder = PrimitiveCylinder(identifier,
                                               self.device)
        self.default_cylinder = PrimitiveGaussian(identifier,
                                                  'Cylinder', primitive_cylinder,
                                                  config=config, is_trainable=True)

        self.cached_output = None

    def to(self, device):
        super().to(device)
        self.default_cylinder.to(device)

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        if not len(self.in_edges) and self.cached_output is not None:
            inputs_dict[self.name][NodeStrings.OUT_str + 'Mesh'] = \
                self.cached_output[NodeStrings.OUT_str + 'Mesh']
            return

        assert torch.all(inputs_dict[self.name][NodeStrings.IN_str +
                                                NodeMeshPrimitiveCylinderStrings.Vertices_str] == 32)
        assert torch.all(inputs_dict[self.name][NodeStrings.IN_str +
                                                NodeMeshPrimitiveCylinderStrings.Side_Segments_str] == 1)
        assert torch.all(inputs_dict[self.name][NodeStrings.IN_str +
                                                NodeMeshPrimitiveCylinderStrings.Fill_Segments_str] == 1)

        cylinder_primitive = self.default_cylinder.clone()

        # radius * 2, depth
        radius2 = 2 * \
                  inputs_dict[self.name][NodeStrings.IN_str +
                                         NodeMeshPrimitiveCylinderStrings.Radius_str][:, None]
        depth = inputs_dict[self.name][NodeStrings.IN_str +
                                       NodeMeshPrimitiveCylinderStrings.Depth_str][:, None]
        size = torch.tensor([[[radius2, radius2, depth]]], device=cylinder_primitive.get_device())
        cylinder_primitive.scale(size)

        inputs_dict[self.name][NodeStrings.OUT_str + 'Mesh'] = [cylinder_primitive]

        if not len(self.in_edges):
            self.cached_output = {NodeStrings.OUT_str + 'Mesh': [cylinder_primitive]}

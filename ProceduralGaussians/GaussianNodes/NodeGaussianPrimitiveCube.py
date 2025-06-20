from ProceduralGaussians.GaussianNodes.PrimitiveGaussian import PrimitiveGaussian
from PytorchGeoNodes.Nodes.Node import *
from PytorchGeoNodes.Nodes.NodeMeshPrimitiveCube import PrimitiveCube


class NodeMeshPrimitiveCubeStrings:
    Size_str = 'Size'
    Vertices_X_str = 'Vertices X'
    Vertices_Y_str = 'Vertices Y'
    Vertices_Z_str = 'Vertices Z'


class NodeGaussianPrimitiveCube(Node):
    def __init__(self, identifier, bpy_node, config):
        """
        The Cube node generates a cuboid mesh with variable side lengths and subdivisions. The inside of the mesh is
        still hollow like a normal cube.

        Inputs:
            -- Counts: Number of vertices on the line.
            -- Start Location: Position of the first vertex. (X, Y, Z)
            -- Offset: Offset of the vertices along the line. (X, Y, Z)

        Outputs: Vector

        :param bpy_node:
        """
        super().__init__(bpy_node, config)
        print('Creating MeshPrimitiveCube')

        # TODO Initialize Trainable Gaussian
        primitive_cube = PrimitiveCube(identifier,
                                       primitive_filename='cube_16.obj', device=self.device) # NOTE: Why not use the primitive that was generated for mesh?
        self.default_cube = PrimitiveGaussian(identifier, 'Cube', primitive_cube,
                                              config=config, is_trainable=True)

        #print("DEFAULT CUBE: ", self.default_cube.get_scales().shape)
        self.cached_output = None

    def to(self, device):
        super().to(device)

        self.default_cube.to(device)

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        if not len(self.in_edges) and self.cached_output is not None:
            inputs_dict[self.name][NodeStrings.OUT_str + 'Mesh'] = \
                self.cached_output[NodeStrings.OUT_str + 'Mesh']
            return

        assert torch.all(inputs_dict[self.name][NodeStrings.IN_str +
                                                NodeMeshPrimitiveCubeStrings.Vertices_X_str] == 2)
        assert torch.all(inputs_dict[self.name][NodeStrings.IN_str +
                                                NodeMeshPrimitiveCubeStrings.Vertices_Y_str] == 2)
        assert torch.all(inputs_dict[self.name][NodeStrings.IN_str +
                                                NodeMeshPrimitiveCubeStrings.Vertices_Z_str] == 2)

        size = inputs_dict[self.name][NodeStrings.IN_str +
                                      NodeMeshPrimitiveCubeStrings.Size_str]

        cube_primitive = self.default_cube.clone()
        assert cube_primitive.get_device() == size.device
        assert size.shape[0] == 1, "Batch size > 1 is not supported yet."
        cube_primitive.scale(size)

        inputs_dict[self.name][NodeStrings.OUT_str + 'Mesh'] = [cube_primitive]

        if not len(self.in_edges):
            self.cached_output = {NodeStrings.OUT_str + 'Mesh': [cube_primitive]}

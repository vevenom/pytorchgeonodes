
from PytorchGeoNodes.Nodes.Node import *
#from PytorchGeoNodes.Nodes.PrimitiveMesh import PrimitiveMesh, join_meshes
from ProceduralGaussians.GaussianNodes.PrimitiveGaussian import join_geometries

class NodeJoinGeometryStrings:
    Geometry_str = 'Geometry'

class NodeGaussianJoinGeometry(Node):
    def __init__(self, bpy_node, config):
        """
        The Join Geometry node merges separately generated geometries into a single one. If the geometry inputs contain
         different types of data, the output will also contain different data types.

        :param bpy_node:
        """
        super().__init__(bpy_node, config)
        print('Creating NodeJoinGeometry')
        self.__type__ = 'NodeJoinGeometry'

        self.cached_output = None

    def forward(self, inputs_dict):
        if not len(self.in_edges) and self.cached_output is not None:
            inputs_dict[self.name][NodeStrings.OUT_str + NodeJoinGeometryStrings.Geometry_str] = \
                self.cached_output[NodeStrings.OUT_str + NodeJoinGeometryStrings.Geometry_str]
            return

        inputs_dict[self.name] = {}

        inputs_dict[self.name][NodeStrings.IN_str + NodeJoinGeometryStrings.Geometry_str] = []
        for edge in self.in_edges:

            from_node = edge.from_node
            # to_node = edge.to_node
            from_socket = edge.from_socket
            to_socket = edge.to_socket

            assert to_socket.identifier == NodeJoinGeometryStrings.Geometry_str, \
                f'Socket {to_socket.identifier} not supported for NodeJoinGeometry.'

            # Outer list is for batch dimension, inner list is for meshes to join
            if not inputs_dict[self.name][NodeStrings.IN_str +
                                          NodeJoinGeometryStrings.Geometry_str]:
                inputs_dict[self.name][NodeStrings.IN_str +
                                       NodeJoinGeometryStrings.Geometry_str] = \
                    [[v] for v in inputs_dict[from_node.name][NodeStrings.OUT_str + from_socket.identifier]
                     if not v.is_empty()]
            else:
                for i, v in enumerate(inputs_dict[from_node.name][NodeStrings.OUT_str +
                                                                  from_socket.identifier]):
                    if not v.is_empty():
                        inputs_dict[self.name][NodeStrings.IN_str +
                                               NodeJoinGeometryStrings.Geometry_str][i].append(v)

        inputs_dict[self.name][NodeStrings.OUT_str +
                               NodeJoinGeometryStrings.Geometry_str] = []
        for i, v in enumerate(inputs_dict[self.name][NodeStrings.IN_str +
                                                     NodeJoinGeometryStrings.Geometry_str]):
            inputs_dict[self.name][NodeStrings.OUT_str +
                                   NodeJoinGeometryStrings.Geometry_str].append(join_geometries(v))

        if not len(self.in_edges):
            self.cached_output = {NodeStrings.OUT_str +
                                  NodeJoinGeometryStrings.Geometry_str: inputs_dict[self.name][NodeStrings.OUT_str + NodeJoinGeometryStrings.Geometry_str]}

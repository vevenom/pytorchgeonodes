from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Translate, Scale, Transform3d, euler_angles_to_matrix

from PytorchGeoNodes.Nodes.Node import *

class NodeRealizeInstancesStrings:
    GEOMETRY_str = 'Geometry'

class NodeRealizeInstances(Node):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Realize Instances node makes any instances (efficient duplicates of the same geometry) into real geometry
        data. This makes it possible to affect each instance individually, whereas without this node, the exact same
        changes are applied to every instance of the same geometry. However, performance can become much worse when
        the input contains many instances of complex geometry, which is a fundamental limitation when procedurally
        processing geometry.

        :param bpy_node:
        """
        super().__init__(bpy_node)
        print('Creating NodeRealizeInstances')

        self.cached_output = None

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        if not len(self.in_edges) and self.cached_output is not None:
            inputs_dict[self.name][NodeStrings.OUT_str +
                                   NodeRealizeInstancesStrings.GEOMETRY_str] = \
                self.cached_output[NodeStrings.OUT_str + NodeRealizeInstancesStrings.GEOMETRY_str]
            return

        meshes = []
        for mesh_i, mesh in enumerate(inputs_dict[self.name][NodeStrings.IN_str +
                                                             NodeRealizeInstancesStrings.GEOMETRY_str]):
            new_mesh = Meshes(verts=[mesh.verts_packed()], faces=[mesh.faces_packed()])
            meshes.append(new_mesh)

        inputs_dict[self.name][NodeStrings.OUT_str + 'Geometry'] = meshes

        if not len(self.in_edges):
            self.cached_output = {NodeStrings.OUT_str + 'Geometry': meshes}


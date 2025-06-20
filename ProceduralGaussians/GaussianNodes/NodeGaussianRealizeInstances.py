from pytorch3d.structures import Meshes
from pytorch3d.transforms import Rotate, Translate, Scale, Transform3d, euler_angles_to_matrix

from PytorchGeoNodes.Nodes.Node import *

class NodeRealizeInstancesStrings:
    GEOMETRY_str = 'Geometry'

class NodeGaussianRealizeInstances(Node):
    def __init__(self, bpy_node, config):
        """
        The Realize Instances node makes any instances (efficient duplicates of the same geometry) into real geometry
        data. This makes it possible to affect each instance individually, whereas without this node, the exact same
        changes are applied to every instance of the same geometry. However, performance can become much worse when
        the input contains many instances of complex geometry, which is a fundamental limitation when procedurally
        processing geometry.

        :param bpy_node:
        """
        super().__init__(bpy_node, config)
        print('Creating NodeRealizeInstances')

        self.cached_output = None

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        if not len(self.in_edges) and self.cached_output is not None:
            inputs_dict[self.name][NodeStrings.OUT_str +
                                   NodeRealizeInstancesStrings.GEOMETRY_str] = \
                self.cached_output[NodeStrings.OUT_str + NodeRealizeInstancesStrings.GEOMETRY_str]
            return

        primitive_gaussians = []
        for pg_i, pg in enumerate(inputs_dict[self.name][NodeStrings.IN_str +
                                                             NodeRealizeInstancesStrings.GEOMETRY_str]):
            new_pg = pg.clone()
            primitive_gaussians.append(new_pg)

            assert new_pg.batch_size == pg.batch_size, f'got {new_pg.batch_size} mesh vertices'

        inputs_dict[self.name][NodeStrings.OUT_str + 'Geometry'] = primitive_gaussians

        if not len(self.in_edges):
            self.cached_output = {NodeStrings.OUT_str + 'Geometry': primitive_gaussians}


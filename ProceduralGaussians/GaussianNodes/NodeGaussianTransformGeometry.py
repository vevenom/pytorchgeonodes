from pytorch3d.transforms import Scale, Transform3d, euler_angles_to_matrix

from PytorchGeoNodes.Nodes.Node import *
#from PytorchGeoNodes.Nodes.PrimitiveMesh import PrimitiveMesh
from ProceduralGaussians.GaussianNodes.PrimitiveGaussian import PrimitiveGaussian

class NodeTransformGeometryStrings:
    Geometry_str = 'Geometry'
    Translation_str = 'Translation'
    Rotation_str = 'Rotation'
    Scale_str = 'Scale'


class NodeGaussianTransformGeometry(Node):
    def __init__(self, bpy_node, config):
        """
        The Transform Geometry Node allows you to move, rotate or scale the geometry. The transformation is applied to
        the entire geometry, and not per element. The Set Position Node is used for moving individual points of a
        geometry. For transforming instances individually, the instance translate, rotate, or scale nodes can be used.

        :param bpy_node:
        """
        super().__init__(bpy_node, config)
        print('Creating NodeTransformGeometry')

        self.cached_output = None

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        geometry = inputs_dict[self.name][NodeStrings.IN_str +
                                          NodeTransformGeometryStrings.Geometry_str] # should be a list of mesh
        translation = inputs_dict[self.name][NodeStrings.IN_str +
                                             NodeTransformGeometryStrings.Translation_str]
        translation = torch.zeros_like(translation) - translation
        translation[..., 0] = -translation[..., 0]  # invert y and z axis

        rotation = inputs_dict[self.name][NodeStrings.IN_str +
                                          NodeTransformGeometryStrings.Rotation_str]
        scale = inputs_dict[self.name][NodeStrings.IN_str +
                                       NodeTransformGeometryStrings.Scale_str]

        transformed_gaussians = []

        assert translation.shape[0] == 1, "Batch size > 1 is not supported yet."
        assert rotation.shape[0] == 1, "Batch size > 1 is not supported yet."
        assert scale.shape[0] == 1, "Batch size > 1 is not supported yet."
        for pg_i, pg in enumerate(geometry):
            if pg.is_empty():
                transformed_gaussians.append(pg)
                continue

            assert isinstance(pg, PrimitiveGaussian), (
                'Geometry must be a PrimitiveGaussian, got {}'.format(type(pg)))

            translation_i = translation[pg_i]

            # rotation_i is in Euler angles, convert to rotation matrix
            rotation_i = euler_angles_to_matrix(rotation[pg_i], convention='XYZ')
            scale_i = scale[pg_i]

            # create transform matrix from translation, rotation and scale
            transform_matrix = torch.eye(4, device=self.device)
            transform_matrix = transform_matrix[None].expand(1, transform_matrix.shape[0], transform_matrix.shape[1])
            transform_matrix[:, 3, :3] = translation_i
            transform_matrix[:, :3, :3] = rotation_i

            # NOTE: scale_mat seems to be not used !!!!
            scale_mat = torch.eye(4, device=self.device)
            scale_mat = scale_mat.view(1, 4, 4).expand(1, 4, 4)
            scale_mat[:, 0, 0] = scale_i[:, 0]
            scale_mat[:, 1, 1] = scale_i[:, 1]
            scale_mat[:, 2, 1] = scale_i[:, 2]

            scale = Scale(scale_i)
            transform3d = scale.compose(Transform3d(matrix=transform_matrix))
            if pg.primitive_type == 'Empty':
                continue

            # pg = pg.clone()
            pg = pg.transform(transform3d)
            transformed_gaussians.append(pg)

        inputs_dict[self.name][NodeStrings.OUT_str + 'Geometry'] = transformed_gaussians


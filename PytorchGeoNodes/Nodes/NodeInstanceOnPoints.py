from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.transforms import Rotate, Translate, Scale, Transform3d, euler_angles_to_matrix

from PytorchGeoNodes.Nodes.Node import *
from PytorchGeoNodes.Nodes.PrimitiveMesh import PrimitiveMesh, join_cloned_verts_faces

class NodeInstanceOnPointsString:
    Points_str = 'Points'
    Selection_str = 'Selection'
    Pick_Instance_str = 'Pick Instance'
    Instance_Index_str = 'Instance Index'
    Instance_str = 'Instance'
    Rotation_str = 'Rotation'
    Scale_str = 'Scale'

class NodeInstanceOnPoints(Node):
    def __init__(self, bpy_node, config):
        """
        The Instance on Points node adds a reference to a geometry to each of the points present in the input geometry.
        Instances are a fast way to add the same geometry to a scene many times without duplicating the underlying data.
        The node works on any geometry type with a Point domain, including meshes, point clouds, and curve
        control points.

        Any attributes on the points from the Geometry input will be available on the instance domain of the
        generated instances.

        :param bpy_node:
        """
        super().__init__(bpy_node, config)
        print('Creating NodeInstanceOnPoints')

        self.cached_output = None

        assert not bpy_node.inputs['Pick Instance'].default_value, 'Pick Instance not supported yet'

    def forward(self, inputs_dict):
        super().forward(inputs_dict)  # Changes inputs_dict in place

        if not len(self.in_edges) and self.cached_output is not None:
            inputs_dict[self.name][NodeStrings.OUT_str + 'Instances'] = \
                self.cached_output[NodeStrings.OUT_str + 'Instances']
            return

        points = inputs_dict[self.name][NodeStrings.IN_str + NodeInstanceOnPointsString.Points_str]

        instance = inputs_dict[self.name][NodeStrings.IN_str + NodeInstanceOnPointsString.Instance_str] # should be a mesh
        selection = inputs_dict[self.name][NodeStrings.IN_str + 'Selection']
        rotation = inputs_dict[self.name][NodeStrings.IN_str + NodeInstanceOnPointsString.Rotation_str]
        scale = inputs_dict[self.name][NodeStrings.IN_str + NodeInstanceOnPointsString.Scale_str]

        pick_instance = inputs_dict[self.name][NodeStrings.IN_str + NodeInstanceOnPointsString.Pick_Instance_str]
        instance_index = inputs_dict[self.name][NodeStrings.IN_str + NodeInstanceOnPointsString.Instance_Index_str]

        assert torch.all(selection == 1), 'Selection = False not supported yet'
        assert torch.all(pick_instance == False), 'Pick = True not supported yet'
        assert torch.all(instance_index == 0), 'Instance Index != 0 not supported yet'

        # copy instance for each point
        meshes = []

        for mesh_i, mesh in enumerate(instance):
            assert isinstance(mesh, PrimitiveMesh), 'Geometry must be a PrimitiveMesh, got {}'.format(type(mesh))
            verts = mesh.verts
            faces = mesh.faces

            assert verts.shape[0] == mesh.verts.shape[0], 'Batch size > 1 not supported yet'

            rotation_i = rotation[mesh_i]
            scale_i = scale[mesh_i]

            # Batched version
            points_i = torch.zeros_like(points[mesh_i]) - points[mesh_i]
            points_i[..., 0] = -points_i[..., 0]

            v = verts.expand(points_i.shape[0], verts.shape[1], verts.shape[2])
            r = rotation_i.expand(points_i.shape[0], rotation_i.shape[1])
            s = scale_i.expand(points_i.shape[0], scale_i.shape[1])

            r = euler_angles_to_matrix(r, convention='XYZ')

            translation_tf = Translate(points_i)
            rotation_tf = Rotate(r)
            scale_tf = Scale(s)

            transform = Transform3d().compose(scale_tf, rotation_tf, translation_tf).to(self.device)
            v = transform.transform_points(v)

            # faces = faces[None].expand(points_i.shape[0], faces.shape[1], faces.shape[2])
            # mesh_p = mesh.clone()
            # mesh_p.verts = v
            # mesh_p.faces = faces
            mesh_p = join_cloned_verts_faces(v, faces, mesh, 'InstanceOnPoints')

            meshes.append(mesh_p)

        inputs_dict[self.name][NodeStrings.OUT_str + 'Instances'] = meshes

        if not len(self.in_edges):
            self.cached_output = {NodeStrings.OUT_str + 'Instances': meshes}


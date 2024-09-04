from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.transforms import Rotate, Translate, Scale, Transform3d, euler_angles_to_matrix

from PytorchGeoNodes.Nodes.Node import *

class NodeInstanceOnPointsString:
    Points_str = 'Points'
    Selection_str = 'Selection'
    Pick_Instance_str = 'Pick Instance'
    Instance_Index_str = 'Instance Index'
    Instance_str = 'Instance'
    Rotation_str = 'Rotation'
    Scale_str = 'Scale'

class NodeInstanceOnPoints(Node):
    def __init__(self, bpy_node: bpy.types.GeometryNode):
        """
        The Instance on Points node adds a reference to a geometry to each of the points present in the input geometry.
        Instances are a fast way to add the same geometry to a scene many times without duplicating the underlying data.
        The node works on any geometry type with a Point domain, including meshes, point clouds, and curve
        control points.

        Any attributes on the points from the Geometry input will be available on the instance domain of the
        generated instances.

        :param bpy_node:
        """
        super().__init__(bpy_node)
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

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # copy instance for each point
        meshes = []

        start.record()

        for mesh_i, mesh in enumerate(instance):
            assert isinstance(mesh, Meshes), 'Instance must be a mesh'

            verts = mesh.verts_packed()
            faces = mesh.faces_packed()

            rotation_i = rotation[mesh_i]
            scale_i = scale[mesh_i]

            # Batched version
            points_i = torch.zeros_like(points[mesh_i]) - points[mesh_i]
            points_i[..., 0] = -points_i[..., 0]

            v = verts.expand(points_i.shape[0], verts.shape[0], verts.shape[1])
            r = rotation_i.expand(points_i.shape[0], rotation_i.shape[1])
            s = scale_i.expand(points_i.shape[0], scale_i.shape[1])

            r = euler_angles_to_matrix(r, convention='XYZ')

            translation_tf = Translate(points_i)
            rotation_tf = Rotate(r)
            scale_tf = Scale(s)

            transform = Transform3d().compose(scale_tf, rotation_tf, translation_tf).to(self.device)
            v = transform.transform_points(v)

            faces = faces[None].expand(points_i.shape[0], faces.shape[0], faces.shape[1])

            mesh_p = Meshes(verts=v, faces=faces)

            meshes.append(mesh_p)
        end.record()

        torch.cuda.synchronize()
        # if start.elapsed_time(end) > 3:
        # print('**********Time needed for instance on points: {}'.format(start.elapsed_time(end)))

        inputs_dict[self.name][NodeStrings.OUT_str + 'Instances'] = meshes

        if not len(self.in_edges):
            self.cached_output = {NodeStrings.OUT_str + 'Instances': meshes}


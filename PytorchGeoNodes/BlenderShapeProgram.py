import open3d as o3d
import torch
import numpy as np
import bpy
import os
import sys
import json
from contextlib import contextmanager
import time

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesVertex

from PytorchGeoNodes.ShapeParamsTree.ShapeParamsTree import ShapeParamsTree

class BlenderShapeProgram(object):
    def __init__(self, config_path, tmp_obj_name='tmp_sp_obj.obj', write_sleep=0.05):
        self.tmp_obj_path = os.path.join('/tmp/', tmp_obj_name)

        # Wait when exporting files to avoid bad file descriptors
        self.write_sleep = write_sleep

        with open(config_path, 'r') as f:
            self.config = json.load(f)
            print("Loaded config for class %s" % self.config['class_name'])

        self.blender_obj = self._get_obj_()
        self._modifiers_dict = self._map_named_params_to_modifier_ids()

    def get_modifier_value(self, modifier_name):
        return self._modifiers_dict[modifier_name]

    def parse_params_tree_(self):
        """
        Parses the parameters tree

        :return: parameters tree
        :rtype: ShapeParamsTree
        """

        params_tree = ShapeParamsTree()
        params_tree.build_tree_from_config(self.config)

        return params_tree


    def _generate_params_types_dict(self):
        """
        Generates a dictionary of parameter types
        :return: a dictionary of parameter types
        """
        params_types_dict = {}
        params_config = self.config['params']
        for param_name in params_config.keys():
            if params_config[param_name]['type'] == 'float':
                params_types_dict[param_name] = np.float
            elif params_config[param_name]['type'] == 'int':
                params_types_dict[param_name] = np.int
            elif params_config[param_name]['type'] == 'bool':
                params_types_dict[param_name] = np.bool
            else:
                raise NotImplementedError("Parameter type %s not implemented" % params_config[param_name]['type'])

        return params_types_dict

    def _generate_params_values_dict(self):
        """
        Generates a dictionary of allowed parameter value values/range
        :return: a dictionary of allowed parameter values/ranges
        """
        params_values_dict = {}
        params_config = self.config['params']
        for param_name in params_config.keys():
            if 'range' in params_config[param_name].keys():
                assert self.params_types_dict[param_name] == np.float or self.params_types_dict[param_name] == np.int, \
                    "'range' is only supported for float and int types"
                if self.params_types_dict[param_name] == np.float:
                    params_values_dict[param_name] = params_config[param_name]['range']
                else:
                    param_range = params_config[param_name]['range']
                    params_values_dict[param_name] = set(range(param_range[0], param_range[1]))
            elif 'values' in params_config[param_name].keys():
                assert self.params_types_dict[param_name] == np.int, \
                    " 'values' is only supported for int type"
                params_values_dict[param_name] = set(params_config[param_name]['values'])
            elif self.params_types_dict[param_name] == np.bool:
                params_values_dict[param_name] = {0, 1}
            else:
                raise NotImplementedError("range/values must be defined for %s" % param_name)


        return params_values_dict

    @contextmanager
    def stdout_redirected(self, to=os.devnull):
        '''
        Reference (Accessed 5th June 2023):
         https://blender.stackexchange.com/questions/44560/how-to-supress-bpy-render-messages-in-terminal-output

        import os

        with stdout_redirected(to=filename):
            print("from Python")
            os.system("echo non-Python applications are also supported")
        '''
        fd = sys.stdout.fileno()

        ##### assert that Python and C stdio write using the same file descriptor
        ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

        def _redirect_stdout(to):
            sys.stdout.close()  # + implicit flush()
            os.dup2(to.fileno(), fd)  # fd writes to 'to' file
            sys.stdout = os.fdopen(fd, 'w')  # Python writes to fd

        with os.fdopen(os.dup(fd), 'w') as old_stdout:
            with open(to, 'w') as file:
                _redirect_stdout(to=file)
            try:
                yield  # allow code to be run with the redirected stdout
            finally:
                _redirect_stdout(to=old_stdout)  # restore stdout.
                # buffering and flags such as
                # CLOEXEC may be different

    def _get_obj_(self):
        """
        Loads the object from the blend file

        :return: Blender object
        """

        with self.stdout_redirected():
            program_path = self.config['blend_path']
            bpy.ops.wm.open_mainfile(filepath=program_path)

        assert len(bpy.data.objects) == 1, "Number of objects in the scene is %d != 1" % len(bpy.data.objects)

        print("Loaded object %s" % bpy.data.objects[0].name)

        return bpy.data.objects[0]

    def print_named_params_to_modifiers_table(self):
        node_tree = bpy.data.node_groups['Geometry Nodes']
        modifier = self.blender_obj.modifiers["GeometryNodes"]

        print("{:^90}".format("----- Print named params to modifiers id table -----"))
        format = "| {:^30} | {:^30} | {:^30} |"
        print(format.format('Name', 'Modifier id', 'Value'))
        for param_name in node_tree.inputs.keys():
            if param_name == 'Geometry':
                continue
            param_id = node_tree.inputs[param_name].name
            print(format.format(param_name, param_id, str(modifier[param_id])))
    def _map_named_params_to_modifier_ids(self):
        """
        :return: a dictionary of named parameters and their corresponding modifier ids
        """
        node_tree = bpy.data.node_groups['Geometry Nodes']

        modifiers_dict = {}
        # node_tree.inputs was replaced by node_tree.interface.items_tree in bpy 4.0
        for param_name in node_tree.interface.items_tree.keys():
            if param_name == 'Geometry':
                continue
            param_id = node_tree.interface.items_tree[param_name].identifier
            param_name = node_tree.interface.items_tree[param_name].name
            modifiers_dict[param_name] = param_id

        return modifiers_dict

    def get_params_dict(self):
        """
        :return: a dictionary of parameters and their values
        """
        param_values_dict = {}
        for param_name in self._modifiers_dict.keys():
            param_value = \
                bpy.data.objects[self.blender_obj.name].modifiers["GeometryNodes"][self._modifiers_dict[param_name]]
            param_values_dict[param_name] = param_value

        return param_values_dict

    def get_params_list(self):
        """
        :return: a list of parameter values
        """

        param_values_list = []
        for param_name in self._modifiers_dict.keys():
            param_value = \
                bpy.data.objects[self.blender_obj.name].modifiers["GeometryNodes"][self._modifiers_dict[param_name]]
            param_values_list.append(param_value)

        return param_values_list

    def print_params(self):
        """
        Print params dict to console
        :return:
        """

        print("{:^60}".format("----- Print named params and values -----"))
        format = "{:^30} | {:^30}"
        print(format.format('Name', 'Value'))
        for param_name in self.get_params_dict():
            param_value = \
                bpy.data.objects[self.blender_obj.name].modifiers["GeometryNodes"][self._modifiers_dict[param_name]]
            print(format.format(param_name, str(param_value)))

    def set_params(self, params_dict):
        """
        Set parameters of the program
        :param params_dict: a dictionary of parameters and their values
        :return:
        """

        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = self.blender_obj
        self.blender_obj.select_set(True)

        for param_name in params_dict.keys():
            type_a = \
                type(
                    bpy.data.objects[self.blender_obj.name].modifiers["GeometryNodes"][self._modifiers_dict[param_name]]
                )
            type_b = type(params_dict[param_name])

            assert type_b == type_a, "Type of param %s is %s != %s" % (param_name, type_a, type_b)

            bpy.data.objects[self.blender_obj.name].modifiers["GeometryNodes"][self._modifiers_dict[param_name]] = \
                params_dict[param_name]

    def set_params_from_tree(self, params_tree: ShapeParamsTree):
        """
        Set parameters of the program from a tree

        :param params_tree: a tree of parameters and their values
        :type params_tree: ShapeParamsTree
        :return:
        """

        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = self.blender_obj
        self.blender_obj.select_set(True)

        params_nodes = params_tree.get_leaf_nodes()
        for node in params_nodes:
            param_name = node.name
            value_modifier = \
                bpy.data.objects[self.blender_obj.name].modifiers["GeometryNodes"][self._modifiers_dict[param_name]]

            assert isinstance(value_modifier, node.type), "Type of param %s is %s != %s" % \
                                                   (param_name, type(value_modifier), node.type)

            bpy.data.objects[self.blender_obj.name].modifiers["GeometryNodes"][self._modifiers_dict[param_name]] = \
                node.value

    def export_as_obj(self, output_path):
        """
            Export the program as an obj
            :param output_path: the path to save the obj file
            :return:
        """
        with self.stdout_redirected():

            copied_obj = self.blender_obj.copy()

            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.collection.objects.link(copied_obj)
            bpy.context.view_layer.objects.active = copied_obj
            copied_obj.select_set(True)

            bpy.ops.object.convert(target='MESH', keep_original=False)
            bpy.ops.object.modifier_add(type='TRIANGULATE')
            # bpy.ops.export_scene.obj(filepath=output_path, use_selection=True)
            bpy.ops.wm.obj_export(filepath=output_path, export_selected_objects=True)

            bpy.ops.object.delete()

            bpy.ops.object.select_all(action='DESELECT')
            bpy.context.view_layer.objects.active = self.blender_obj
            self.blender_obj.select_set(True)


        # Wait to avoid bad file descriptors
        time.sleep(self.write_sleep)

    def o3d_get_mesh(self, compute_normals=True, load_texture=False):
        self.export_as_obj(self.tmp_obj_path)
        mesh = o3d.io.read_triangle_mesh(self.tmp_obj_path)
        if compute_normals:
            mesh.compute_triangle_normals()

        verts = mesh.vertices

        return mesh

    def py3d_get_mesh(self, load_texture=False):
        mesh = self.o3d_get_mesh(compute_normals=False)

        verts = torch.from_numpy(np.asarray(mesh.vertices)).to(torch.float32)
        faces = torch.from_numpy(np.asarray(mesh.triangles))

        if load_texture:
            if mesh.has_textures():
                raise NotImplementedError("Loading textures is not implemented yet")

            mesh = Meshes(verts=[verts],
                          faces=[faces],
                          textures=TexturesVertex(verts_features=torch.zeros_like(verts)[None] + 0.5))
        else:
            mesh = Meshes(verts=[verts], faces=[faces])

        return mesh

    def o3d_sample_point_cloud(self, num_points):
        mesh = self.o3d_get_mesh(compute_normals=False)
        # mesh = mesh.simplify_vertex_clustering(0.1)

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(mesh.vertices)

        point_cloud += mesh.sample_points_uniformly(num_points)

        return point_cloud

    def visualize_o3d(self):
        """
        Visualize the program using open3d
        :return:
        """
        mesh = self.o3d_get_mesh(compute_normals=True)
        o3d.visualization.draw_geometries([mesh])



# if __name__ == '__main__':
#     shape_program = BlenderShapeProgram(config_path='../configs_shape_programs/shape_program_cabinet.json')
#
#
#     shape_program.params_tree.randomize_tree_values()
#     shape_program.set_params_from_tree(shape_program.params_tree)
#     shape_program.params_tree.visualize_tree(show_values=True)
#     assert False

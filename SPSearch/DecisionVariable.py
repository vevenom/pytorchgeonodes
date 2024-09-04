import numpy as np
import torch
import torch.nn as nn

from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import Transform3d
from pytorch3d.structures import Meshes

from PytorchGeoNodes.ShapeParamsTree.ShapeParamsTree import ShapeParamsTree


class DecisionValue(object):
    def __init__(self, dv_name, value):
        self.dv_name = dv_name
        self.value = value

    def __str__(self):
        # return "DecisionValue: {0} visited {1} times".format(self.value, self.visits)
        return 'val_' + str(self.value.detach().cpu().numpy().flatten()[0])

    def get_value(self):
        return self.value


class DecisionVariable(object):
    def __init__(self, name, valid_range=None, or_dependencies=None, not_dependencies=None,
                 add_value=False, normalized_values=False):
        self.name = name
        self.values = None  # torch.tensor([])

        self.valid_range = valid_range

        self.or_dependencies = or_dependencies
        self.not_dependencies = not_dependencies

        self.add_value = add_value

        self.normalized_values = normalized_values

    def __str__(self):
        return "DecisionVariable: {0} = {1}".format(self.name, self.values)

    def get_name(self):
        return self.name

    def init_values(self, values):
        self.values = values

    def get_values(self):
        return self.values

    def get_value(self, ind):
        return self.values[ind]

    def convert_value(self, value):
        value = value.get_value()
        if self.normalized_values:
            # values are in [-1, 1] range
            value = (value + 1) / 2.0 * (self.valid_range[1] - self.valid_range[0]) + self.valid_range[0]
        return value

    @staticmethod
    def generate_dec_vars_from_params_tree(params_tree: ShapeParamsTree, device,
                                           step_size=0.1, cluster_num=4,
                                           normalize_params=True, add_rotation_nodes=True):

        dec_vars = []
        meta_dict = params_tree.get_nodes_meta()

        if add_rotation_nodes:
            # rotation_y = torch.tensor([0.0, np.pi * 0.5, np.pi, np.pi * 1.5], device=device)
            rotation_y = torch.tensor(
                [0.0, np.pi * 0.25, np.pi * 0.5, np.pi * 0.75, np.pi, np.pi * 1.25, np.pi * 1.5], device=device)


            dv_rot = RotationDecisionVariable('OBJ_Rotation', valid_range=[-np.pi / 4, 2 * np.pi],
                                      normalized_values=normalize_params)

            # normalize values (if you don't want them normalized set dv_rot.normalized_values = False in constructor)
            rotation_y = (rotation_y + np.pi / 4) / (2 * np.pi + np.pi / 4) * 2 - 1

            values = []
            for value in rotation_y:
                values.append(DecisionValue(dv_name=dv_rot.name, value=nn.Parameter(value[None, None])))

            dv_rot.init_values(values)
            dec_vars.append(dv_rot)

        for param_name, param_meta in meta_dict.items():
            or_dependencies = param_meta['or_dependencies']
            not_dependencies = param_meta['not_dependencies']

            values = []
            if param_meta['type'] == float:
                valid_range = param_meta['valid_range']

                # create values for decision variable based on the valid range and linspace steps
                # step_size = 0.05
                values_torch = (
                    torch.tensor(np.arange(valid_range[0] + step_size / 2,
                                           valid_range[1] - step_size / 2 + 1e-4, step_size),
                                            dtype=torch.float32, device=device))


                dec_var = DecisionVariable(param_name,  valid_range=valid_range,
                                           or_dependencies=or_dependencies, not_dependencies=not_dependencies,
                                           normalized_values=normalize_params)

                if values_torch.shape[0] <= 2:
                    # if number of values is <= 2, then we initialize with the middle value of the valid range
                    values_torch = torch.tensor([valid_range[0] + (valid_range[1] - valid_range[0]) / 2.0],
                                                device=device)

                if normalize_params:
                    values_torch = (values_torch - valid_range[0]) / (valid_range[1] - valid_range[0]) * 2 - 1

                for value in values_torch:
                    values.append(DecisionValue(dv_name=param_name, value=nn.Parameter(value[None, None])))

            elif param_meta['type'] == bool:
                dec_var = DecisionVariable(param_name,
                                           or_dependencies=or_dependencies,
                                           not_dependencies=not_dependencies)
                values_torch = torch.tensor([False, True], device=device)
                for value in values_torch:
                    values.append(DecisionValue(dv_name=param_name, value=value[None, None]))

            elif param_meta['type'] == int:
                dec_var = DecisionVariable(param_name,
                                           or_dependencies=or_dependencies,
                                           not_dependencies=not_dependencies)
                valid_values = param_meta['valid_values']
                valid_values = list(valid_values)
                values_torch = torch.tensor(valid_values, device=device)
                for value in values_torch:
                    values.append(DecisionValue(dv_name=param_name, value=value[None, None]))
            else:
                raise Exception(f'Param type {param_meta["type"]} not supported yet.')

            dec_var.init_values(values)
            dec_vars.append(dec_var)

        return dec_vars

    @staticmethod
    @torch.no_grad()
    def order_dv_based_on_geometry_variance(
            sp_tree, decision_variables_list, geometry_nodes, init_shapes_n=100):
        """
        Order decision variables based on variance of geometry nodes

        :param sp_tree:
        :type sp_tree: ShapeParamsTree
        :param decision_variables_list: list of decision variables
        :type decision_variables_list: [DecisionVariable]
        :param geometry_nodes:
        :type geometry_nodes: GeometryNodes
        :param init_shapes_n:
        :return:
        """

        init_input_params_dict_list = []
        for init_sp_ind in range(init_shapes_n):
            sp_tree.randomize_tree_values()
            init_shape_params = sp_tree.to_params_dict()

            init_input_params_dict_list.append(init_shape_params)

        device = decision_variables_list[0].values[0].value.device

        # Calculate variance of geometry nodes for each decision variable
        dv_variance_list = []
        for dv in decision_variables_list:
            print('Calculating variance for decision variable:', dv.name)
            print([dv.convert_value(value)[0, 0].item() for value in dv.values])

            # chamfer_dists = torch.zeros(len(dv.values), device=device)
            chamfer_dists = torch.zeros((len(init_input_params_dict_list), len(dv.values)), device=device)

            if len(dv.values) == 1:
                dv_variance_list.append(0)
                continue

            for shape_ind, init_input_params in enumerate(init_input_params_dict_list):

                # convert to tensor
                input_params_dict = {}
                for dv1 in decision_variables_list:
                    if not isinstance(dv1, RotationDecisionVariable):
                        input_params_dict[dv1.name] = torch.tensor([[init_input_params[dv1.name]]], device=device)

                # get initial mesh
                _, outputs = geometry_nodes.forward(input_params_dict, transform2blender_coords=True)
                init_obj_mesh = outputs[0][0][0]

                # get initial points and normals
                init_pcd, init_normals = (
                    sample_points_from_meshes(init_obj_mesh, num_samples=10000, return_normals=True))

                # go through all values of the decision variable and calculate chamfer distance to initial mesh
                for value_ind, value in enumerate(dv.values):

                    if not isinstance(dv, RotationDecisionVariable):
                        input_params_dict[dv.name] = value.value

                    _, outputs = geometry_nodes.forward(input_params_dict, transform2blender_coords=True)
                    obj_mesh = outputs[0][0][0]

                    if isinstance(dv, RotationDecisionVariable):
                        verts = obj_mesh.verts_packed()
                        faces = obj_mesh.faces_packed()

                        rotation_matrix = dv.get_rotation_matrix_from_y_angle(dv.convert_value(value))

                        transform = Transform3d(matrix=rotation_matrix, device=device)
                        assert rotation_matrix.shape[0] == 1

                        verts = transform.transform_points(verts)
                        obj_mesh = Meshes(verts=[verts], faces=[faces], textures=obj_mesh.textures)

                    pcd, normals = (
                        sample_points_from_meshes(obj_mesh, num_samples=10000, return_normals=True))

                    cd_points, cd_normals = chamfer_distance(init_pcd, pcd, x_normals=init_normals,
                                                             y_normals=normals)

                    chamfer_dists[shape_ind][value_ind] = cd_points

            dv_variance_list.append(
                ((chamfer_dists - chamfer_dists.mean(dim=1, keepdims=True)) ** 2).sum(dim=1).mean().item()
            )

        for dv_ind, dv in enumerate(decision_variables_list):
            print(dv.name, dv_variance_list[dv_ind])

        # Sort decision variables based on variance
        decision_variables_list = [
            x for _, x in sorted(zip(dv_variance_list, decision_variables_list),
                                 key=lambda pair: pair[0], reverse=True)]

        for dv1_ind in range(len(decision_variables_list)):
            dv1 = decision_variables_list[dv1_ind]

            if dv1.or_dependencies is not None:
                or_dependencies = [d if not isinstance(d, list) else d[0] for d in dv1.or_dependencies]
                for dv2_ind, dv2 in enumerate(decision_variables_list[dv1_ind + 1:]):
                    if dv2.name in or_dependencies:
                        print(dv2.name, dv1.name)

                        decision_variables_list.remove(dv2)
                        decision_variables_list.insert(dv1_ind, dv2)

                        break
            elif dv1.not_dependencies is not None:
                for dv2_ind, dv2 in enumerate(decision_variables_list[dv1_ind + 1:]):
                    if dv2.name in dv1.not_dependencies:
                        decision_variables_list.remove(dv2)
                        decision_variables_list.insert(dv1_ind, dv2)
                        break

        print("Sorted decision variables based on variance and dependencies:")
        for dv_ind, dv in enumerate(decision_variables_list):
            print('--', dv.name)

        return decision_variables_list


class RotationDecisionVariable(DecisionVariable):
    def __init__(self, name, valid_range=None, add_value=False, normalized_values=False):
        super(RotationDecisionVariable, self).__init__(name,
                                                       valid_range=valid_range,
                                                       add_value=add_value,
                                                       normalized_values=normalized_values)

    def get_rotation_matrix_from_y_angle(self, angle_y_rad):

        angle_tensor = torch.zeros_like(angle_y_rad)
        angle_tensor = angle_tensor.repeat(*angle_y_rad.shape[:-1], 3)
        angle_tensor[:, 1] = angle_y_rad

        # create rotation matrix
        # create eye matrix
        rot_matrix = torch.eye(4, device=angle_tensor.device)
        rot_matrix = rot_matrix[None].repeat(*angle_tensor.shape[:-1], 1, 1)
        rot_matrix[:, :3, :3] = axis_angle_to_matrix(angle_tensor)

        return rot_matrix


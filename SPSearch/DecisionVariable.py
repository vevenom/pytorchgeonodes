import numpy as np
import torch
import torch.nn as nn

from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import Transform3d
from pytorch3d.structures import Meshes

from PytorchGeoNodes.ShapeParamsTree.ShapeParamsTree import ShapeParamsTree

class DecisionValuePriors:
    def __init__(self, bbox_size_mean, bbox_size_std):
        self.bbox_size_mean = bbox_size_mean
        self.bbox_size_cov = bbox_size_std

    def compute_prior_distribution_from_pcd(self, target_pcd, epsilon=1e-3):
        assert target_pcd.shape[0] == 1

        target_bb_min = target_pcd.min(dim=1)[0]
        target_bb_max = target_pcd.max(dim=1)[0]
        target_bb_size = torch.abs(target_bb_max - target_bb_min)

        gaussian_distr = (
            torch.distributions.multivariate_normal.MultivariateNormal(self.bbox_size_mean, self.bbox_size_cov))
        prior_distribution = torch.exp(gaussian_distr.log_prob(target_bb_size))

        prior_distribution = prior_distribution / prior_distribution.sum()

        return prior_distribution

        # mean_dist = (target_bb_size - self.bbox_size_mean)
        # pdf_value = torch.bmm(mean_dist[:, None], torch.linalg.inv(self.bbox_size_cov))
        # pdf_value = torch.bmm(pdf_value, mean_dist[..., None])
        # pdf_value = torch.exp(-0.5 * pdf_value)[:,0,0]

        # pdf_constant = (
        #         1.0 / ((2 * np.pi) ** (3 / 2.0) * torch.linalg.det(self.bbox_size_cov)) ** 0.5)
        # pdf_value = pdf_constant * torch.exp(pdf_value)[:,0,0]

        # pdf_value =  pdf_value[:, 0, 0]

        # print(pdf_value)
        #
        # # prior_distribution = torch.exp(pdf_value) / torch.sum(torch.exp(pdf_value + epsilon))
        # prior_distribution = torch.nn.functional.softmax(pdf_value, dim=0)
        #
        # return prior_distribution

class DecisionValue(torch.nn.Module):
    def __init__(self, dv_name, value):
        super().__init__()
        self.dv_name = dv_name
        self.value = value

        # MCTS
        self.max_score = -np.inf
        self.visits = 0
        self.min_score = np.inf
        self.score_sum = 0
        self.mov_avg = None
        self.avg_decay = 0.9

    def __str__(self):
        # return "DecisionValue: {0} visited {1} times".format(self.value, self.visits)
        return 'val_' + str(self.value.detach().cpu().numpy().flatten()[0])

    def get_value(self):
        return self.value

    # MCTS
    def get_score(self, mode):
        if mode == 'AVG':
            return self.score_sum / float(self.visits) if self.visits > 0 else -np.inf
        elif mode == 'MAX':
            return self.max_score
        elif mode == 'MIN':
            return self.min_score
        elif mode == 'MOV_AVG':
            return self.mov_avg
        else:
            raise Exception("Unknown mode {0}".format(mode))

    def get_visits(self):
        return self.visits

    def update(self, score, n_visits):
        if score > self.max_score:
            self.max_score = score
        if score < self.min_score:
            self.min_score = score
        self.score_sum += score
        self.visits += n_visits

        if self.mov_avg is None:
            self.mov_avg = score
        else:
            self.mov_avg = self.avg_decay * self.mov_avg + (1 - self.avg_decay) * score

class DecisionVariable(torch.nn.Module):
    def __init__(self, name, valid_range=None, or_dependencies=None, not_dependencies=None,
                 add_value=False, normalized_values=False):
        super().__init__()
        self.name = name
        self.values = None

        self.valid_range = valid_range

        self.or_dependencies = or_dependencies
        self.not_dependencies = not_dependencies

        self.add_value = add_value

        self.normalized_values = normalized_values

        self.prior = None # type: DecisionValuePriors

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

    def share_memory_(self):
        for value in self.values:
            value.value.share_memory_()

    def set_prior(self, prior):
        self.prior = prior

    @staticmethod
    def generate_dec_vars_from_params_tree(params_tree: ShapeParamsTree, device,
                                           step_size=0.2, cluster_num=4,
                                           normalize_params=True, add_rotation_nodes=True):

        dec_vars = []
        meta_dict = params_tree.get_nodes_meta()

        if add_rotation_nodes:
            # rotation_y = torch.tensor([0.0, np.pi * 0.5, np.pi, np.pi * 1.5], device=device)
            # rotation_y = torch.tensor(
            #     [0.0, np.pi * 0.25, np.pi * 0.5, np.pi * 0.75, np.pi, np.pi * 1.25, np.pi * 1.5], device=device)

            rotation_y = torch.tensor(
                [0.0, np.pi * 0.5, np.pi, np.pi * 1.5], device=device)


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
                # values_torch = (
                #     torch.tensor(np.arange(valid_range[0] + step_size / 2,
                #                            valid_range[1] - step_size / 2 + 1e-4, step_size),
                #                             dtype=torch.float32, device=device))

                values_torch = (
                    torch.tensor(np.arange(valid_range[0],
                                           valid_range[1] + 1e-4, step_size),
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
    def preprocess_dv(
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
            bbox_sizes = torch.zeros((len(init_input_params_dict_list), len(dv.values), 3), device=device)

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
                init_obj_mesh = Meshes(verts=init_obj_mesh.verts, faces=init_obj_mesh.faces)

                # get initial points and normals
                init_pcd, init_normals = (
                    sample_points_from_meshes(init_obj_mesh, num_samples=10000, return_normals=True))

                # go through all values of the decision variable and calculate chamfer distance to initial mesh
                for value_ind, value in enumerate(dv.values):

                    if not isinstance(dv, RotationDecisionVariable):
                        input_params_dict[dv.name] = dv.convert_value(value)

                    _, outputs = geometry_nodes.forward(input_params_dict, transform2blender_coords=True)
                    obj_mesh = outputs[0][0][0]
                    obj_mesh = Meshes(verts=obj_mesh.verts, faces=obj_mesh.faces)

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

                    mesh_bb = obj_mesh.get_bounding_boxes()  # (N, 3, 2)
                    mesh_size = torch.abs(mesh_bb[0,:,1] - mesh_bb[0,:,0])
                    bbox_sizes[shape_ind][value_ind] = mesh_size

            dv_variance_list.append(
                ((chamfer_dists - chamfer_dists.mean(dim=1, keepdims=True)) ** 2).sum(dim=1).mean().item()
            )

            bbox_mean = bbox_sizes.mean(dim=[0])
            diffs = (bbox_sizes - bbox_mean[None]).reshape(-1, 3)
            bbox_cov = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(init_shapes_n, len(dv.values), 3, 3)
            bbox_cov = bbox_cov.sum(dim=0) / (len(dv.values) - 1)

            # Add small value to variance to ensure it is positive definite
            bbox_cov[:, 0, 0] += 1e-1
            bbox_cov[:, 1, 1] += 1e-1
            bbox_cov[:, 2, 2] += 1e-1
            # print("Bbox mean:", bbox_mean)
            # print("Bbox cov:", bbox_cov)

            prior = DecisionValuePriors(bbox_mean, bbox_cov)
            dv.set_prior(prior)

        for dv_ind, dv in enumerate(decision_variables_list):
            print(dv.name, "CD variance", dv_variance_list[dv_ind])

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


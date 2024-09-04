import numpy as np
import torch
import yaml
import json
import argparse
import os
import shutil
from pytorch3d.io import load_obj, save_obj

from PytorchGeoNodes.GeometryNodes import GeometryNodes
from PytorchGeoNodes.BlenderShapeProgram import BlenderShapeProgram

from utils import DictAsMember


def calculate_mesh_from_input_dict(self, input_params_dict, rotation_matrix, translation_offset=None):

    from pytorch3d.transforms import Transform3d, Translate
    from pytorch3d.structures import Meshes

    device = self.device

    _, outputs = self.geometry_nodes.forward(input_params_dict, transform2blender_coords=True)
    obj_mesh = outputs[0][0][0]

    verts = obj_mesh.verts_packed()
    faces = obj_mesh.faces_packed()

    transform = Transform3d(device=device)

    if rotation_matrix is not None:
        transform_rot = Transform3d(matrix=rotation_matrix, device=device)
        assert rotation_matrix.shape[0] == 1
        transform = transform.compose(transform_rot)
    else:
        assert False, "We are doing experiments with rotations"

    # transform = transform.compose(Transform3d(matrix=T_scan2CAD))

    # If object is not at origin
    bb = obj_mesh.get_bounding_boxes()  # (N, 3, 2)

    bb_center = (bb[:, :, 1] - bb[:, :, 0]) / 2 + bb[:, :, 0]

    verts = verts - bb_center

    mesh = Meshes(verts=[verts], faces=[faces], textures=obj_mesh.textures)
    bb = mesh.get_bounding_boxes()  # (N, 3, 2)

    bb_center = (bb[:, :, 1] - bb[:, :, 0]) / 2 + bb[:, :, 0]

    verts = transform.transform_points(verts)

    obj_mesh = Meshes(verts=[verts], faces=[faces], textures=obj_mesh.textures)

    bb_center = transform.transform_points(bb_center)

    translation = self.obj_center + bb_center
    verts = verts + translation

    if translation_offset is not None:
        translate = Translate(translation_offset)
        verts = translate.transform_points(verts)

    obj_mesh = Meshes(verts=[verts], faces=[faces], textures=obj_mesh.textures)

    return obj_mesh


def parse_params_from_json(json_params):
    """
    Parse parameters from json file
    """

    from pytorch3d.transforms import axis_angle_to_matrix

    # Load parameters from json file
    input_params_dict = {}
    angle_x_rad = None
    for key, value in json_params['input_dict'].items():
        if key not in ['OBJ_Rotation', 'translation_offset']:
            input_params_dict[key] = torch.tensor([[value]], device=device)
    angle_x_rad = torch.tensor([[json_params['rotation_angle_y']]], device=device)
    translation_offset = torch.tensor(json_params['translation_offset'], device=device)

    angle_tensor = torch.zeros_like(angle_x_rad)
    angle_tensor = angle_tensor.repeat(*angle_x_rad.shape[:-1], 3)
    angle_tensor[:, 1] = angle_x_rad

    # create rotation matrix
    # create eye matrix
    rot_matrix = torch.eye(4, device=angle_tensor.device)
    rot_matrix = rot_matrix[None].repeat(*angle_tensor.shape[:-1], 1, 1)
    rot_matrix[:, :3, :3] = axis_angle_to_matrix(angle_tensor)

    return input_params_dict, rot_matrix, translation_offset

if __name__ == '__main__':

    # parse input args
    # -- category
    # -- server / local

    parser = argparse.ArgumentParser(description='Reconstruct scannotate objects')
    parser.add_argument('--category', type=str, default='cabinet', help='Object category')
    parser.add_argument('--experiments_path', type=str, help='Experiment name')
    parser.add_argument('--experiment_name', type=str, help='Experiment name')
    parser.add_argument('--solution_name', type=str, help='Solution name to get')
    # 0best_0_solution.json, final_solution.json

    args = parser.parse_args()

    object_category = args.category
    experiments_path = args.experiments_path
    experiment_name = args.experiment_name
    solution_name = args.solution_name

    shape_program = BlenderShapeProgram(config_path='configs_shape_programs/sp_' + object_category + '.json')
    params_tree = shape_program.parse_params_tree_()

    # set cuda if available
    device = torch.device("cuda:0")

    geometry_nodes = GeometryNodes(shape_program)
    geometry_nodes.to(device)

    # experiment_name = 'synthetic_reconstructions'
    experiment_path = os.path.join(experiments_path, experiment_name, object_category)

    reconstructed_meshes_path = os.path.join(experiments_path, 'reconstructed_meshes')
    reconstructed_meshes_path = os.path.join(reconstructed_meshes_path, experiment_name, object_category)

    scenes_names = os.listdir(experiment_path)
    scenes_names.sort()

    for scene_name in scenes_names:
        if scene_name == 'settings.json':
            continue
        scene_reconstructions_path = os.path.join(experiment_path, scene_name)

        out_folder_path = os.path.join(reconstructed_meshes_path, scene_name)
        os.makedirs(out_folder_path, exist_ok=True)

        reconstructed_obj_mesh_path = (
            os.path.join(out_folder_path, 'reconstructed_mesh.obj'))
        transformation_params_path = os.path.join(out_folder_path,
                                                  'transformation_params.json')

        # final_src_solution_path = os.path.join(scene_reconstructions_path, '0final_0_solution.json')
        final_src_solution_path = os.path.join(scene_reconstructions_path, solution_name)
        with open(final_src_solution_path, 'r') as f:
            final_json = json.load(f)

        final_dst_solution_path = os.path.join(out_folder_path, 'final_solution.json')
        shutil.copy(final_src_solution_path, final_dst_solution_path)

        input_params_dict, rot_matrix, translation_offset = parse_params_from_json(final_json)

        # obj_mesh = calculate_mesh_from_input_dict(input_params_dict, rot_matrix, translation_offset)

        _, outputs = geometry_nodes.forward(input_params_dict, transform2blender_coords=True)
        obj_mesh = outputs[0][0][0]

        # Save mesh
        save_obj(reconstructed_obj_mesh_path, obj_mesh.verts_packed(), obj_mesh.faces_packed())

        del final_json["input_dict"]

        # Save json
        with open(transformation_params_path, 'w') as f:
            json.dump(final_json, f)




        # except Exception as e:
        #     print(f"Error in scene {scene_name}: {e}")
        #     continue













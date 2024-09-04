import argparse
import copy
import os
import sys
import yaml

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)

import pickle
import torch
import json

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_gather, knn_points, sample_points_from_meshes
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes

from utils import DictAsMember

from SPSearch.DecisionVariable import DecisionVariable
from SPSearch.SyntheticTarget.SyntheticTarget import SyntheticTarget
from PytorchGeoNodes.BlenderShapeProgram import BlenderShapeProgram
from PytorchGeoNodes.GeometryNodes import GeometryNodes

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
    translation_offset = translation_offset[None]

    # angle_x_degree = angle_x_rad * 180.0 / np.pi
    angle_tensor = torch.zeros_like(angle_x_rad)
    angle_tensor = angle_tensor.repeat(*angle_x_rad.shape[:-1], 3)
    angle_tensor[:, 1] = angle_x_rad

    # create rotation matrix
    # create eye matrix
    rot_matrix = torch.eye(4, device=angle_tensor.device)
    rot_matrix = rot_matrix[None].repeat(*angle_tensor.shape[:-1], 1, 1)
    rot_matrix[:, :3, :3] = axis_angle_to_matrix(angle_tensor)

    return input_params_dict, rot_matrix, translation_offset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reconstruct scannotate objects')
    parser.add_argument('--experiments_path', type=str, help='Experiment path')
    parser.add_argument('--experiment_name', type=str, help='Experiment name')
    parser.add_argument('--category', type=str, help='Category name')
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    args = parser.parse_args()

    obj_category = args.category
    experiments_path = args.experiments_path
    dataset_path = args.dataset_path
    dataset_path = os.path.join(dataset_path, obj_category)

    if obj_category == 'cabinet':
        shape_program = BlenderShapeProgram(config_path='configs_shape_programs/sp_synth_cabinet.json')
    else:
        shape_program = BlenderShapeProgram(config_path='configs_shape_programs/sp_' + obj_category + '.json')
    params_tree = shape_program.parse_params_tree_()

    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    experiment_folder = os.path.join(experiments_path, 'reconstructed_meshes/',
                                     args.experiment_name, obj_category)
    scenes = os.listdir(experiment_folder)
    scenes.sort()

    geometry_nodes = GeometryNodes(shape_program)
    geometry_nodes.to(device)

    avg_chamfer_dist = 0.0
    avg_loss = 0.0
    num_scenes = len(scenes)
    for scene in scenes:
        if scene == 'settings.json':
            continue
        print("Evaluating scene: ", scene)
        scene_path = os.path.join(experiment_folder, scene)
        annotation_path = os.path.join(dataset_path, scene, 'scene_dict.pkl')

        # Load scene object
        scene_obj_path = os.path.join(scene_path, 'reconstructed_mesh.obj')

        pred_json_path = os.path.join(scene_path, 'final_solution.json')
        with open(pred_json_path, 'r') as f:
            predictions_json = json.load(f)


        annotation_dict = pickle.load(open(annotation_path, 'rb'))
        gt_points = annotation_dict['scene_pcd']

        num_samples_points = gt_points.shape[1]

        # Load scene mesh
        pred_scene_mesh = load_obj(scene_obj_path, load_textures=False)
        verts, faces = pred_scene_mesh[0], pred_scene_mesh[1].verts_idx
        pred_scene_mesh = Meshes(verts=[verts], faces=[faces])

        pred_points = sample_points_from_meshes(pred_scene_mesh, num_samples_points, return_normals=False)
        pred_points = pred_points.to(device)

        target = SyntheticTarget(annotation_dict,
                                 geometry_nodes,
                                 log_path='/tmp/tmp_synthetic_target')


        decision_variables = DecisionVariable.generate_dec_vars_from_params_tree(params_tree, device)


        with torch.no_grad():
            input_params_dict, rot_matrix, translation_offset = parse_params_from_json(predictions_json)
            loss = target.calculate_cost_from_input_dict(input_params_dict,
                                                         rotation_matrix=rot_matrix,
                                                         translation_offset=translation_offset)

        avg_loss += loss
        # calculate chamfer distance
        chamfer_dist, _ = chamfer_distance(pred_points, gt_points, )
        avg_chamfer_dist += chamfer_dist

        print('Chamfer distance: ', chamfer_dist)
        print('Loss: ', loss)

    avg_chamfer_dist = avg_chamfer_dist / num_scenes
    avg_loss = avg_loss / num_scenes
    print(f'Average chamfer distance for experiment {args.experiment_name} and '
          f'category {obj_category} is {avg_chamfer_dist}')
    print(f'Average loss for experiment {args.experiment_name} and '
            f'category {obj_category} is {avg_loss}')



import numpy as np
import torch
import yaml
import os
import pickle
import shutil
import argparse

from PytorchGeoNodes.GeometryNodes import GeometryNodes
from PytorchGeoNodes.BlenderShapeProgram import BlenderShapeProgram

from SPSearch.DecisionVariable import DecisionVariable
from utils import DictAsMember, set_seed

from eval_helpers import *

set_seed(seed=3407)

skip_existing_reconstructions = True

device = torch.device("cuda:0")

if __name__ == '__main__':

    # parse input args
    # -- category
    # -- server / local

    parser = argparse.ArgumentParser(description='Reconstruct scannotate objects')
    parser.add_argument('--category', type=str, default='cabinet', help='Object category')
    parser.add_argument('--synthetic_dataset_path', type=str,
                        help='Dataset path')
    parser.add_argument('--experiments_path', type=str,
                        help='Experiments path')
    parser.add_argument('--experiment_name',
                        type=str, help='Name of the experiment')
    parser.add_argument('--solution_name',
                        type=str, help='Solution file name')

    args = parser.parse_args()

    object_category = args.category

    general_config_path = 'configs/general_config.yaml'
    with open(general_config_path, 'r') as f:
        general_config = yaml.load(f, Loader=yaml.FullLoader)
    general_config = DictAsMember(general_config)

    synthetic_dataset_path = args.synthetic_dataset_path
    synthetic_dataset_path = os.path.join(general_config.experiments_path_base, synthetic_dataset_path, object_category)

    reconstruction_annotations_path = args.experiments_path
    reconstruction_annotations_experiment_name = args.experiment_name
    reconstruction_annotations_path = os.path.join(general_config.experiments_path_base,
                                                   reconstruction_annotations_path,
                                                   reconstruction_annotations_experiment_name,
                                                   object_category)
    reconstruction_annotations_file_name = args.solution_name



    shape_program = BlenderShapeProgram(config_path='configs_shape_programs/sp_' + object_category + '.json')
    params_tree = shape_program.parse_params_tree_()

    scannotate_config_path = general_config.scannotate_config_path
    # scannotate_config_path = 'data_config/scannotate_config.yaml'os.path.join(
    with open(scannotate_config_path, 'r') as f:
        scannotate_config = yaml.load(f, Loader=yaml.FullLoader)
    scannotate_config = DictAsMember(scannotate_config)

    scenes_names = os.listdir(synthetic_dataset_path)
    scenes_names.sort()

    settings_path = 'configs/genetic_settings.yaml'
    with open(settings_path, 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    settings = DictAsMember(settings)

    if 'load_ordered_dv' in settings.keys() and settings.load_ordered_dv:
        processed_data_path = os.path.join(general_config.experiments_path_base,
                                           general_config.processed_data_path)
        ordered_dv_path = os.path.join(processed_data_path, object_category + '_ord_dv.pickle')
        with open(ordered_dv_path, 'rb') as f:
            decision_variables = pickle.load(f)
    else:
        decision_variables = DecisionVariable.generate_dec_vars_from_params_tree(params_tree, device)

    scores_dicts = {}
    sample_n_dicts = {}

    for scene_name in scenes_names:

        scene_dict_path = os.path.join(synthetic_dataset_path, scene_name, 'scene_dict.pkl')
        with open(scene_dict_path, 'rb') as f:
            scene_dict = pickle.load(f)
        gt_params = scene_dict['sp_params']

        print(reconstruction_annotations_path, scene_name, reconstruction_annotations_file_name)
        reconstruction_obj_json_path = os.path.join(
            reconstruction_annotations_path,
            scene_name,
            reconstruction_annotations_file_name
        )

        if not os.path.exists(reconstruction_obj_json_path):
            print('[Warning] : {} does not exist. Skipping ....'.format(reconstruction_obj_json_path))
            continue

        if object_category == 'table':
            result = evaluate_table(reconstruction_obj_json_path, '', gt_params=gt_params)
        elif object_category == 'sofa':
            result = evaluate_sofa(reconstruction_obj_json_path, '', gt_params=gt_params)
        elif object_category == 'chair':
            result = evaluate_chair(reconstruction_obj_json_path, '', gt_params=gt_params)
        elif object_category == 'cabinet':
            result = evaluate_cabinet(reconstruction_obj_json_path, '', gt_params=gt_params)

        for key in result.keys():
            if key in scores_dicts.keys():
                scores_dicts[key] += result[key]
                sample_n_dicts[key] += 1
            else:
                scores_dicts[key] = result[key]
                sample_n_dicts[key] = 1

    for key in scores_dicts.keys():
        scores_dicts[key] /= sample_n_dicts[key]
        scores_dicts[key] = np.round(scores_dicts[key], 3)

    # Split into classification and mse
    for key in scores_dicts.keys():
        print(key, scores_dicts[key])

    results_path = os.path.join(reconstruction_annotations_path, 'sp_metrics.json')
    print('Write results to json: ', results_path)
    with open(results_path, 'w') as f:
        json.dump(scores_dicts, f)

    # assert False
    # except Exception as e:
    #     print(f"Error in scene {scene_name}: {e}")
    #     continue

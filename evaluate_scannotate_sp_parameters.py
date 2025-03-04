import numpy as np
import torch
import yaml
import os
import pickle
import argparse

from ScanNetAnnotation import *

from PytorchGeoNodes.BlenderShapeProgram import BlenderShapeProgram

from SPSearch.DecisionVariable import DecisionVariable
from utils import DictAsMember, set_seed
from SPSearch.utils import parse_params_from_json

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
    parser.add_argument('--annotations_path', type=str,
                        default='./sp_gt_annotations/',
                        help='Dataset path')
    parser.add_argument('--annotations_file_name',
                        default='sp_params.json',
                        type=str, help='Dataset path')
    parser.add_argument('--experiments_path', type=str,
                        help='Dataset path')
    parser.add_argument('--experiment_name',
                        type=str, help='Dataset path')
    parser.add_argument('--solution_name',
                        default='final_solution.json',
                        type=str, help='Dataset path')

    args = parser.parse_args()

    object_category = args.category

    manual_annotations_path = args.annotations_path
    manual_annotations_path = os.path.join(manual_annotations_path, object_category)
    manual_annotations_file_name = args.annotations_file_name

    reconstruction_annotations_file_name = args.solution_name

    general_config_path = 'configs/general_config.yaml'
    with open(general_config_path, 'r') as f:
        general_config = yaml.load(f, Loader=yaml.FullLoader)
    general_config = DictAsMember(general_config)

    reconstruction_annotations_path = os.path.join(general_config.experiments_path_base, args.experiments_path)
    reconstruction_annotations_experiment = args.experiment_name
    reconstruction_annotations_path = os.path.join(reconstruction_annotations_path,
                                                   reconstruction_annotations_experiment)

    shape_program = BlenderShapeProgram(config_path='configs_shape_programs/sp_' + object_category + '.json')
    params_tree = shape_program.parse_params_tree_()

    scannotate_config_path = general_config.scannotate_config_path
    with open(scannotate_config_path, 'r') as f:
        scannotate_config = yaml.load(f, Loader=yaml.FullLoader)
    scannotate_config = DictAsMember(scannotate_config)

    scenes_names = os.listdir(manual_annotations_path)
    scenes_names.sort()


    valid_scenes_path = os.path.join('./valid_scenes', object_category, 'scenes_objs_dict_to_keep.json')

    with open(valid_scenes_path, 'r') as f:
        valid_scenes_json = json.load(f)

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
        scannotate_annotation_file = (
            os.path.join(scannotate_config.scannotate_masks_path, scene_name, scene_name + '.pkl'))
        # print('scannotate_annotation_file', scannotate_annotation_file)
        if not os.path.exists(scannotate_annotation_file):
            continue
        with open(scannotate_annotation_file, 'rb') as f:
            scannotate_objects = pickle.load(f)  # type: ScanNetAnnotation

        manual_scene_path = os.path.join(manual_annotations_path, scene_name)

        for obj_idx, box_item in enumerate(scannotate_objects.obj_annotation_list):
            obj_id = box_item.object_id

            if box_item.category_label != object_category:
                continue

            recon_obj_id = int(obj_id) - 1
            recon_obj_name = 'obj_' + str(recon_obj_id)

            manual_obj_json_path = os.path.join(manual_scene_path, recon_obj_name, manual_annotations_file_name)
            if not os.path.exists(manual_obj_json_path):
                continue

            with open(manual_obj_json_path, 'r') as f:
                obj_json = json.load(f)

            reconstruction_obj_json_path = os.path.join(
                reconstruction_annotations_path,
                scene_name,
                recon_obj_name,
                reconstruction_annotations_file_name
            )

            if not os.path.exists(reconstruction_obj_json_path):
                print('[Warning] : {} does not exist. Skipping ....'.format(reconstruction_obj_json_path))
                continue

            if object_category == 'table':
                result = evaluate_table(reconstruction_obj_json_path, manual_obj_json_path)
            elif object_category == 'sofa':
                result = evaluate_sofa(reconstruction_obj_json_path, manual_obj_json_path)
            elif object_category == 'chair':
                result = evaluate_chair(reconstruction_obj_json_path, manual_obj_json_path)

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

    results_path = os.path.join(reconstruction_annotations_path, object_category + '_sp_metrics.json')
    print('Write results to json: ', results_path)
    with open(results_path, 'w') as f:
        json.dump(scores_dicts, f)




    # assert False
    # except Exception as e:
    #     print(f"Error in scene {scene_name}: {e}")
    #     continue

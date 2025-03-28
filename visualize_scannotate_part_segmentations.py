import numpy as np
import torch
import yaml
import os
import pickle
import shutil
import argparse
import json
import open3d as o3d

from ScanNetAnnotation import *
from SPSearch.ScannotateTarget.ScannotateTarget import ScannotateTarget

from PytorchGeoNodes.GeometryNodes import GeometryNodes
from PytorchGeoNodes.BlenderShapeProgram import BlenderShapeProgram

from SPSearch.DecisionVariable import DecisionVariable
from utils import DictAsMember, set_seed
from SPSearch.utils import parse_params_from_json

set_seed(seed=3407)

device = torch.device("cuda:0")

if __name__ == '__main__':

    # parse input args
    # -- category
    # -- server / local

    parser = argparse.ArgumentParser(description='Reconstruct scannotate objects')
    parser.add_argument('--solution_name',
                        default='final_solution.json',
                        type=str, help='Dataset path')
    parser.add_argument('--experiments_path', type=str, help='Experiment path')
    parser.add_argument('--experiment_name', type=str, help='Experiment name')

    args = parser.parse_args()

    experiment_name = args.experiment_name

    rotation_err_dict, samples_n_dict, cd_error_dict = {}, {}, {}
    rotation_err_dict[experiment_name] = 0.0
    samples_n_dict[experiment_name] = 0
    cd_error_dict[experiment_name] = 0.0
    for object_category in ['chair', 'table', 'sofa']:
    # for object_category in ['table']:
        print('Evaluate', object_category)
        manual_annotations_path = './sp_gt_annotations'
        manual_annotations_path = os.path.join(manual_annotations_path, object_category)
        manual_annotations_file_name = 'sp_params.json'

        reconstruction_annotations_path = args.experiments_path
        reconstruction_annotations_experiment = experiment_name
        reconstruction_annotations_path = os.path.join(reconstruction_annotations_path,
                                                       reconstruction_annotations_experiment)
        reconstruction_annotations_file_name = args.solution_name

        general_config_path = 'configs/general_config.yaml'
        with open(general_config_path, 'r') as f:
            general_config = yaml.load(f, Loader=yaml.FullLoader)
        general_config = DictAsMember(general_config)

        shape_program = BlenderShapeProgram(config_path='configs_shape_programs/sp_' + object_category + '.json')
        params_tree = shape_program.parse_params_tree_()

        geometry_nodes = GeometryNodes(shape_program)
        geometry_nodes.to(device)

        scannotate_config_path = general_config.scannotate_config_path
        # scannotate_config_path = 'data_config/scannotate_config.yaml'os.path.join(
        with open(scannotate_config_path, 'r') as f:
            scannotate_config = yaml.load(f, Loader=yaml.FullLoader)
        scannotate_config = DictAsMember(scannotate_config)

        scenes_names = os.listdir(manual_annotations_path)
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

                reconstruction_obj_json_path = os.path.join(
                    reconstruction_annotations_path,
                    scene_name,
                    recon_obj_name,
                    reconstruction_annotations_file_name
                )

                target = ScannotateTarget(scannotate_objects,
                                          scannotate_config, False,
                                          geometry_nodes, scene_name, obj_idx,
                                          use_object_scale=False,
                                          optimize_translation=True,
                                          log_path=None)

                with open(reconstruction_obj_json_path, 'r') as f:
                    reconstruction_json = json.load(f)
                input_params_dict, rot_matrix, translation_offset = parse_params_from_json(reconstruction_json, device)

                # obj_mesh_pred = target.calculate_mesh_from_input_dict(
                #     input_params_dict, rot_matrix, translation_offset[None])

                target.o3d_visualize_part_segmentation(input_params_dict, rot_matrix, translation_offset[None])




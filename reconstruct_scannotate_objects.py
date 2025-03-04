import numpy as np
import torch
import yaml
import os
import pickle
import shutil
import argparse
import json

from ScanNetAnnotation import *

from PytorchGeoNodes.GeometryNodes import GeometryNodes
from PytorchGeoNodes.BlenderShapeProgram import BlenderShapeProgram

from SPSearch.DecisionVariable import DecisionVariable
from SPSearch.ScannotateTarget.ScannotateTarget import ScannotateTarget
from SPSearch.SPGame import SPGame
from SPSearch.SPSearchLogger import SPSearchLogger
from SPSearch.CoordinateDescent.CoordinateDescent import CoordinateDescent
from SPSearch.Genetic.Genetic import Genetic
from SPSearch.Genetic.GeneticLogger import GeneticLogger
from utils import DictAsMember, set_seed

set_seed(seed=3407)

skip_existing_reconstructions = True

manual_annotations_path = 'sp_gt_annotations_scannotate_old/'
experiment_path = './outputs/reconstructions'

device = torch.device("cuda:0")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Reconstruct scannotate objects')
    parser.add_argument('--category', type=str, default='cabinet', help='Object category')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--experiment_path', type=str, help='Experiments path')
    parser.add_argument('--skip_refinement', action='store_true',
                        help='Skip refinement step')
    parser.add_argument('--method', type=str,
                        help='Search method to use',
                        choices=['cd', 'genetic'])
    args = parser.parse_args()

    object_category = args.category

    general_config_path = 'configs/general_config.yaml'
    with open(general_config_path, 'r') as f:
        general_config = yaml.load(f, Loader=yaml.FullLoader)
    general_config = DictAsMember(general_config)

    experiment_path = args.experiment_path
    experiment_path = os.path.join(general_config.experiments_path_base, experiment_path)

    shape_program = BlenderShapeProgram(config_path='configs_shape_programs/sp_' + object_category + '.json')
    params_tree = shape_program.parse_params_tree_()

    gt_path = os.path.join('./sp_gt_annotations', object_category)
    gt_file_name = 'sp_params.json'

    settings = None
    if args.method == 'cd':
        experiment_path = experiment_path + '_cd'
        settings_path = general_config.cd_config_path
        with open(settings_path, 'r') as f:
            settings = yaml.load(f, Loader=yaml.FullLoader)
        settings = DictAsMember(settings)

        if args.skip_refinement:
            settings['refinement']['use_refinement'] = False
            settings['refinement']['optimize_steps'] = 0
        if not settings.refinement.use_refinement:
            experiment_path = experiment_path + '_no_refinement'

    elif args.method == 'genetic':
        experiment_path = experiment_path + '_genetic'

        genetic_settings_path = 'configs/genetic_settings.yaml'
        with open(genetic_settings_path, 'r') as f:
            genetic_settings = yaml.load(f, Loader=yaml.FullLoader)
        genetic_settings = DictAsMember(genetic_settings)
        if args.skip_refinement:
            genetic_settings['refine_every_n_generations'] = 0
            genetic_settings['refinement']['final_optimization_steps'] = 0
            genetic_settings['refinement']['optimize_steps'] = 0

        if genetic_settings.refine_every_n_generations == 0:
            experiment_path = experiment_path + '_no_refinement'

        settings = genetic_settings
    else:
        raise ValueError('Invalid method. Supported methods: mcts, cd, geocode, probabilistic_model, genetic')

    if 'load_ordered_dv' in settings.keys() and settings.load_ordered_dv:
        processed_data_path = os.path.join(general_config.experiments_path_base,
                                           general_config.processed_data_path)
        ordered_dv_path = os.path.join(processed_data_path, object_category + '_ord_dv.pickle')
        with open(ordered_dv_path, 'rb') as f:
            decision_variables = pickle.load(f)
    else:
        decision_variables = DecisionVariable.generate_dec_vars_from_params_tree(params_tree, device)

    scannotate_config_path = general_config.scannotate_config_path
    with open(scannotate_config_path, 'r') as f:
        scannotate_config = yaml.load(f, Loader=yaml.FullLoader)
    scannotate_config = DictAsMember(scannotate_config)

    scenes_names = os.listdir(scannotate_config.scannotate_masks_path)
    scenes_names.sort()

    shape_program = BlenderShapeProgram(config_path='configs_shape_programs/sp_' + object_category + '.json')
    params_tree = shape_program.parse_params_tree_()

    geometry_nodes = GeometryNodes(shape_program)
    geometry_nodes.to(device)

    os.makedirs(experiment_path, exist_ok=True)

    for scene_name in scenes_names:

        annotation_file = os.path.join(scannotate_config.scannotate_masks_path, scene_name, scene_name + '.pkl')
        with open(annotation_file, 'rb') as f:
            scannotate_objects = pickle.load(f)  # type: ScanNetAnnotation

        for obj_idx, box_item in enumerate(scannotate_objects.obj_annotation_list):
            obj_id = box_item.object_id

            if box_item.category_label != object_category:
                continue

            recon_obj_id = int(obj_id) - 1
            recon_obj_name = 'obj_' + str(recon_obj_id)

            gt_obj_path = os.path.join(gt_path, scene_name, recon_obj_name, gt_file_name)
            if not os.path.exists(gt_obj_path):
                continue

            scene_reconstructions_path = os.path.join(experiment_path, scene_name)
            os.makedirs(scene_reconstructions_path, exist_ok=True)

            obj_reconstruction_path = os.path.join(scene_reconstructions_path, f'obj_{obj_idx}')
            if skip_existing_reconstructions and os.path.exists(obj_reconstruction_path):
                continue
            if os.path.exists(obj_reconstruction_path):
                shutil.rmtree(obj_reconstruction_path)
            os.makedirs(obj_reconstruction_path)

            print(f"Reconstructing {obj_idx} for scene {scene_name}...")

            target = ScannotateTarget(scannotate_objects,
                                      scannotate_config, False,
                                      geometry_nodes, scene_name, obj_idx,
                                      optimize_translation=True,
                                      log_path=obj_reconstruction_path)

            game = SPGame([decision_variables, target])
            logger = SPSearchLogger(game, target)
            with open(os.path.join(obj_reconstruction_path, 'settings.json'), 'w') as f:
                json.dump(settings, f)
            if args.method == 'cd':
                cd = CoordinateDescent(game, scene_reconstructions_path,
                                       settings=settings)
                cd.reconstruct_scene(logger)
            elif args.method == 'genetic' or args.method == 'random':
                genetic_logger = GeneticLogger(game, target, settings)
                genetic = Genetic(game, scene_reconstructions_path, settings=settings)
                genetic.reconstruct_scene(genetic_logger)
            else:
                raise ValueError('Invalid method. Supported methods: mcts, cd')

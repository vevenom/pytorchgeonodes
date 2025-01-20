import torch
import yaml
import os
import pickle
import shutil
import argparse
import json

from PytorchGeoNodes.GeometryNodes import GeometryNodes
from PytorchGeoNodes.BlenderShapeProgram import BlenderShapeProgram

from SPSearch.DecisionVariable import DecisionVariable
from SPSearch.SyntheticTarget.SyntheticTarget import SyntheticTarget
from SPSearch.SPGame import SPGame
from SPSearch.SPSearchLogger import SPSearchLogger
from SPSearch.CoordinateDescent.CoordinateDescent import CoordinateDescent
from SPSearch.Genetic.Genetic import Genetic
from SPSearch.Genetic.GeneticLogger import GeneticLogger
from utils import DictAsMember, set_seed

skip_existing_reconstructions = True
set_seed(seed=3407)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruct objects')
    parser.add_argument('--category', type=str, default='cabinet', help='Object category')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--experiment_path', type=str, help='Experiments path')
    parser.add_argument('--skip_refinement', action='store_true',
                        help='Skip refinement step')
    parser.add_argument('--disable_prior', action='store_true',
                        help='Disable prior distribution during genetic search')
    parser.add_argument('--disable_post_tree', action='store_true',
                        help='Disable posterior tree during genetic search')
    parser.add_argument('--method', type=str,
                        help='Search method to use',
                        choices=['cd', 'genetic'])
    args = parser.parse_args()

    dataset_name = args.dataset_name
    object_category = args.category

    general_config_path = 'configs/general_config.yaml'
    with open(general_config_path, 'r') as f:
        general_config = yaml.load(f, Loader=yaml.FullLoader)
    general_config = DictAsMember(general_config)

    experiment_path = args.experiment_path
    experiment_path = os.path.join(general_config.experiments_path_base, experiment_path)

    synthetic_dataset_path = os.path.join(experiment_path,
                                          dataset_name, object_category)

    experiment_path = os.path.join(experiment_path, dataset_name)

    scenes_names = os.listdir(synthetic_dataset_path)
    scenes_names.sort()

    if object_category == 'cabinet':
        shape_program = BlenderShapeProgram(config_path='configs_shape_programs/sp_synth_cabinet.json')
    else:
        shape_program = BlenderShapeProgram(config_path='configs_shape_programs/sp_' + object_category + '.json')
    params_tree = shape_program.parse_params_tree_()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    geometry_nodes = GeometryNodes(shape_program)
    geometry_nodes.to(device)

    decision_variables = DecisionVariable.generate_dec_vars_from_params_tree(params_tree, device)

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
        if args.disable_post_tree:
            genetic_settings['posterior_tree']['use_posterior_tree'] = False
        if args.disable_prior:
            genetic_settings['posterior_tree']['use_prior_distribution'] = False

        if not genetic_settings.posterior_tree.use_prior_distribution:
            experiment_path = experiment_path + '_no_prior'
        if not genetic_settings.posterior_tree.use_posterior_tree:
            experiment_path = experiment_path + '_no_post'

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

    experiment_path = os.path.join(experiment_path, object_category)

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        # write json settings
        with open(os.path.join(experiment_path, 'settings.json'), 'w') as f:
            json.dump(settings, f, indent=4)

    print(f"Synthetic dataset path: {synthetic_dataset_path}")
    print(f"Experiment path: {experiment_path}")

    for scene_name in scenes_names:

        scene_dict_path = os.path.join(synthetic_dataset_path, scene_name, 'scene_dict.pkl')
        with open(scene_dict_path, 'rb') as f:
            scene_dict = pickle.load(f)

        scene_reconstructions_path = os.path.join(experiment_path, scene_name)
        if skip_existing_reconstructions and os.path.exists(scene_reconstructions_path):
            continue
        if os.path.exists(scene_reconstructions_path):
            shutil.rmtree(scene_reconstructions_path)
        os.makedirs(scene_reconstructions_path, exist_ok=True)

        print(f"Reconstructing scene {scene_name}...")

        target = SyntheticTarget(scene_dict,
                                 geometry_nodes,
                                 log_path=scene_reconstructions_path,
                                 settings=settings.target)

        # Create Game
        game = SPGame([decision_variables, target])

        # Create Logger
        logger = SPSearchLogger(game, target)

        if args.method == 'cd':
            cd = CoordinateDescent(game, scene_reconstructions_path,
                                   settings=settings)
            cd.reconstruct_scene(logger)
        elif args.method == 'genetic':
            genetic_logger = GeneticLogger(game, target, settings)
            genetic = Genetic(game, scene_reconstructions_path, settings=genetic_settings)
            genetic.reconstruct_scene(genetic_logger)
        else:
            raise ValueError('Invalid method. Supported methods: mcts, cd')

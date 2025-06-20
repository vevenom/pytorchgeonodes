import torch
import yaml
import os
import pickle
import argparse
import json

from ScanNetAnnotation import *

from ProceduralGaussians.GSGeometryNodes import GeometryNodes
from PytorchGeoNodes.BlenderShapeProgram import BlenderShapeProgram

from SPSearch.DecisionVariable import DecisionVariable
from SPSearch.ScannotateTarget.ScannotateTarget import ScannotateTarget
from SPSearch.SPGame import SPGame
from utils import DictAsMember, set_seed
from ProceduralGaussians.GaussianScanNetScene import SceneModel

set_seed(seed=3407)

device = torch.device("cuda:0")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Reconstruct scannotate objects')
    parser.add_argument('--category', type=str, default='cabinet', help='Object category')
    parser.add_argument('--annotations_path', type=str, help='Dataset path')
    parser.add_argument('--scene_name', type=str, help='ScanNet Scene name')
    parser.add_argument('--obj_name', type=str, help='Object name')
    parser.add_argument('--annotations_file_name', type=str,
                        default='sp_params.json',
                        help='Dataset path')
    parser.add_argument('--experiment_path',
                        type=str,
                        default='../Pytorchgeonodes_experiments/scannotate_experiments/procedural_gaussians/',
                        help='Experiment path')
    args = parser.parse_args()

    # Prepare variables
    object_category = args.category
    annotations_path = args.annotations_path
    annotations_file_name = args.annotations_file_name

    scene_name = args.scene_name
    obj_name = args.obj_name

    general_config_path = 'configs/general_config.yaml'
    with open(general_config_path, 'r') as f:
        general_config = yaml.load(f, Loader=yaml.FullLoader)
    general_config = DictAsMember(general_config)

    experiment_path = args.experiment_path
    experiment_path = os.path.join(general_config.experiments_path_base, experiment_path)

    scannotate_config_path = general_config.scannotate_config_path
    with open(scannotate_config_path, 'r') as f:
        scannotate_config = yaml.load(f, Loader=yaml.FullLoader)
    scannotate_config = DictAsMember(scannotate_config)

    settings_path = 'configs/genetic_settings.yaml'
    with open(settings_path, 'r') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    settings = DictAsMember(settings)

    scannotate_annotation_file = (
        os.path.join(scannotate_config.scannotate_masks_path, scene_name, scene_name + '.pkl'))
    with open(scannotate_annotation_file, 'rb') as f:
        scannotate_objects = pickle.load(f)  # type: ScanNetAnnotation

    scene_path = os.path.join(annotations_path, scene_name)
    #-------------------------------------------------------------------------------------------------------------------

    # Prepare Geometry Nodes
    shape_program = BlenderShapeProgram(config_path='configs_shape_programs/sp_' + object_category + '.json')
    params_tree = shape_program.parse_params_tree_()

    geometry_nodes = GeometryNodes(shape_program, use_gaussians=True,
                                   config_path='./configs/gs_geometry_nodes_config.yaml')
    geometry_nodes.to(device)

    decision_variables = DecisionVariable.generate_dec_vars_from_params_tree(params_tree, device)
    #-------------------------------------------------------------------------------------------------------------------

    # Find the object in Scannotate annotations
    for obj_idx, box_item in enumerate(scannotate_objects.obj_annotation_list):

        obj_id = box_item.object_id

        if box_item.category_label != object_category:
            continue

        recon_obj_id = int(obj_id) - 1
        recon_obj_name = 'obj_' + str(recon_obj_id)
        if recon_obj_name != obj_name:
            continue

        obj_json_path = os.path.join(scene_path, recon_obj_name, annotations_file_name)
        if not os.path.exists(obj_json_path):
            continue

        with open(obj_json_path, 'r') as f:
            obj_json = json.load(f)

        print(f"Running procedural Gaussians splatting for object {recon_obj_name} for scene {scene_name}...")

        target = ScannotateTarget(scannotate_objects,
                                  scannotate_config, False,
                                  geometry_nodes, scene_name, obj_idx,
                                  optimize_translation=True,
                                  log_path=None)

        obj_experiment_path = os.path.join(experiment_path, scene_name, recon_obj_name)
        os.makedirs(obj_experiment_path, exist_ok=True)

        # Create Gaussian Scene Model Instance
        gaussian_scene = SceneModel(scannotate_objects, geometry_nodes, obj_idx, scannotate_config,
                                    scene_name, obj_experiment_path)

        game = SPGame([decision_variables, target])

        gaussian_scene.optimize(obj_json)
        print("Optimization Done !!!")

        # Object found and reconstructed
        break
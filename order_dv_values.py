import numpy as np
import torch
import argparse
import yaml
import os
import pickle

# datastructures

# 3D transformations functions

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, )

from PytorchGeoNodes.GeometryNodes import GeometryNodes
from PytorchGeoNodes.BlenderShapeProgram import BlenderShapeProgram

from SPSearch.DecisionVariable import DecisionVariable
from utils import DictAsMember

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate decision variable tree for a given object category')
    parser.add_argument('--category', type=str, default='cabinet', help='Object category')
    args = parser.parse_args()

    object_category = args.category

    general_config_path = 'configs/general_config.yaml'
    with open(general_config_path, 'r') as f:
        general_config = yaml.load(f, Loader=yaml.FullLoader)
    general_config = DictAsMember(general_config)

    sp_config_path = f'configs_shape_programs/sp_{args.category}.json'
    shape_program = BlenderShapeProgram(config_path=sp_config_path)
    params_tree = shape_program.parse_params_tree_()

    device = torch.device("cuda:0")

    geometry_nodes = GeometryNodes(shape_program)
    geometry_nodes.to(device)

    decision_variables = DecisionVariable.generate_dec_vars_from_params_tree(params_tree, device)

    processed_data_path = general_config.processed_data_path
    ordered_decision_variables = \
        DecisionVariable.order_dv_based_on_geometry_variance(
            params_tree, decision_variables, geometry_nodes)

    dv_ordered_path = os.path.join(processed_data_path, object_category + '_ord_dv.pickle')
    os.makedirs(processed_data_path, exist_ok=True)
    with open(dv_ordered_path, 'wb') as f:
        pickle.dump(ordered_decision_variables, f)













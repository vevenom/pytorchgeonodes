from pytorch3d.renderer.cameras import PerspectiveCameras
import pytorch3d
import argparse
import sys
import os
import shutil
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRasterizer
)


import json
from config import load_config
import cv2

def main(args):
    config = load_config(args.config)['general']
    SCANNOTATE_PATH = os.path.join(config['SCANNOTATE_PATH'], config['out_folder'])

    scene_list = os.listdir(SCANNOTATE_PATH)
    scene_list.sort()

    for scene_cnt,scene_name in enumerate(scene_list):

        scene_name = scene_name

        masks_path = os.path.join(SCANNOTATE_PATH, scene_name)

        all_inst_seg_2d_path = os.path.join(masks_path, 'all_inst_seg_2d')
        if os.path.isdir(all_inst_seg_2d_path):
            shutil.rmtree(all_inst_seg_2d_path)

        mask2d_from_3d_path = os.path.join(masks_path, 'mask2d_from_3d')
        if os.path.isdir(mask2d_from_3d_path):
            shutil.rmtree(mask2d_from_3d_path)

        sam_results_path = os.path.join(masks_path, 'sam_results')
        if os.path.isdir(sam_results_path):
            shutil.rmtree(sam_results_path)

        skeleton_vis_path = os.path.join(masks_path, 'skeleton_vis')
        if os.path.isdir(skeleton_vis_path):
            shutil.rmtree(skeleton_vis_path)

        valid_maps_path = os.path.join(masks_path, 'valid_maps')
        if os.path.isdir(valid_maps_path):
            shutil.rmtree(valid_maps_path)


# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)

parser = argparse.ArgumentParser(description="Data preprocessing for Scan2CAD")
parser.add_argument("--config", type=str,
                    default='config/0_Scannotate_masks.ini',
                    help="Path to configuration file")

if __name__ == "__main__":
    main(parser.parse_args())
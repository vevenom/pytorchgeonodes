# coding: utf-8
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from pytorch3d.renderer.cameras import PerspectiveCameras
import pytorch3d
import argparse
import open3d as o3d
import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRasterizer
)

from ScanNet_renderer.Torch3DRenderer.pytorch3d_rasterizer_custom import MeshRendererViewSelection
from ScanNet_renderer.Torch3DRenderer.SimpleShader import UVsCorrespondenceShader

import pickle
from utils import transform_ScanNet_to_py3D,alignPclMesh
from load_ScanNet_data import load_intrinsics,load_axis_alignment_mat,get_pose_at_idx
import json
from config import load_config
import cv2

parser = argparse.ArgumentParser(description="Data preprocessing for Scan2CAD")
parser.add_argument("--config", type=str,
                    default=os.path.join(parent, 'config/0_Scannotate_masks.ini'),
                    help="Path to configuration file")

def read_json(filename):
    with open(filename, 'r') as infile:
        return json.load(infile)

def view_selection_scannotate(scene_name,inst_label_list,tmesh,SCANNET_base_path,intrinsics,img_scale,max_views,
                   silhouette_thres,pkl_out_path,scene_obj):

    n_views = 1
    meta_file_path = os.path.join(SCANNET_base_path, scene_name, str(scene_name) + '.txt')

    pose_path = os.path.join(SCANNET_base_path,scene_name, 'pose')
    poses_list = os.listdir(pose_path)
    poses_list.sort(key=lambda x: int(x.split('.')[0]))
    img_list = []
    img_path_list = []
    depth_path_list = []
    R_list = []
    T_list = []
    mseg_path_list = []
    frame_id_list = []

    raster_settings = RasterizationSettings(
        image_size=(int(480 * img_scale), int(640 * img_scale)),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=None,
        perspective_correct=True,
        clip_barycentric_coords=False,
        cull_backfaces=False
    )


    for idx, pose_name in enumerate(poses_list):
        print('view selection for {} {} / {}'.format(scene_name, idx, len(poses_list)), end='\r')
        frame_id_str = pose_name.split('.')
        frame_id_str = frame_id_str[0]
        frame_id = str(int(frame_id_str))

        T_nviews = get_pose_at_idx(meta_file_path, pose_path, frame_id, use_axis_aligned=True)

        if T_nviews is None:
            # insert dummy data to lists, to ensure index consistency
            img_list.append(np.ones((int(480*img_scale),int(640*img_scale)))*-1)
            depth_path_list.append('')
            img_path_list.append('')
            mseg_path_list.append('')
            frame_id_list.append('')
            R_list.append(np.zeros((1,3,3),dtype=np.float64))
            T_list.append(np.zeros((1,3),dtype=np.float64))
            continue

        R_world_to_cam = np.asarray(T_nviews)[:, 0:3, 0:3]
        T_world_to_cam = np.asarray(T_nviews)[:, 0:3, 3]

        R_list.append(R_world_to_cam)
        T_list.append(T_world_to_cam)

        R = torch.tensor(R_world_to_cam).to(device)
        T = torch.tensor(T_world_to_cam).to(device)

        px, py = (intrinsics[0, 2] * img_scale), (intrinsics[1, 2] * img_scale)
        principal_point = torch.tensor([px, py])[None].type(torch.FloatTensor).to(device)
        principal_point = principal_point.repeat(n_views, 1)
        fx, fy = ((intrinsics[0, 0] * img_scale)), ((intrinsics[1, 1] * img_scale))
        focal_length = torch.tensor([fx, fy])[None].type(torch.FloatTensor).to(device)
        focal_length = focal_length.repeat(n_views, 1)

        cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            in_ndc=False,
            device=device, T=T, R=R,
            image_size=((int(480 * img_scale), int(640 * img_scale)),))

        renderer = MeshRendererViewSelection(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=UVsCorrespondenceShader(
                device=device,
                cameras=cameras
            )

        )
        tmesh = tmesh.extend(n_views)
        img,fragments = renderer(meshes_world=tmesh.to(device))
        valid_pix = fragments.pix_to_face.repeat(1,1,1,4)

        img[valid_pix < 0] = -1.
        img = img.cpu().detach().numpy()[0,:,:,0]
        img[img < 0.] = 0.
        img = np.round(img * 255.).astype(np.uint8)

        img_list.append(img)
        frame_id_list.append(frame_id)

    img_ary = np.asarray(img_list)

    out_path_mask = os.path.join(pkl_out_path, 'mask2d_from_3d')
    if not os.path.exists(out_path_mask):
        os.makedirs(out_path_mask)

    out_path_inst_seg_2d = os.path.join(pkl_out_path, 'all_inst_seg_2d')
    if not os.path.exists(out_path_inst_seg_2d):
        os.makedirs(out_path_inst_seg_2d)


    #for label in inst_label_list:
    for obj_annotation in scene_obj.obj_annotation_list:
        label = obj_annotation.object_id
        if label == -255:
            continue
        mask = np.zeros_like(img_ary)
        mask[img_ary == label] = 1

        if obj_annotation.object_id != label:
            assert False

        frame_id_targets = obj_annotation.view_params['frame_ids']
        views_ = []
        for frame_id_target in frame_id_targets:
            if frame_id_target == '':
                frame_id_target = '0'
            view = frame_id_list.index(frame_id_target)
            views_.append(view)

        frame_id_ary = np.asarray(frame_id_list)[views_]
        mask_ary_selected = mask[views_]
        inst_seg_selected = img_ary[views_]

        for (frame_id_tmp, mask_tmp, inst_seg_2d) in zip(frame_id_ary,mask_ary_selected,inst_seg_selected):

            if np.sum(inst_seg_2d) < 1:
                assert False, 'Should not happen'

            if np.sum(mask_tmp) < 1:
                assert False, 'Should not happen'

            file_out_path = os.path.join(out_path_mask, str(int(label)) + '_' + str(frame_id_tmp) + '.png')
            cv2.imwrite(file_out_path, mask_tmp * 255)

            file_out_path = os.path.join(out_path_inst_seg_2d, str(int(label)) + '_' + str(frame_id_tmp) + '.png')
            cv2.imwrite(file_out_path,inst_seg_2d)

            inst_seg_2d_vis = inst_seg_2d / np.max(inst_seg_2d)
            inst_seg_2d_vis*= 255
            file_out_path = os.path.join(out_path_inst_seg_2d, str(int(label)) + '_' + str(frame_id_tmp) + '_vis_.png')
            cv2.imwrite(file_out_path,inst_seg_2d_vis)

    print('\n')
    return None

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def main(args):

    config = load_config(args.config)['general']
    SCANNOTATE_PATH = config['SCANNOTATE_PATH']
    SCANNET_base_path = config['SCANNET_base_path']

    # text_file = open(os.path.join(SCANNET_PATH,'ScanNet_splits','scannetv2_' + data_split + '.txt'), "r")
    # scene_list = text_file.readlines()
    scene_list = os.listdir(SCANNET_base_path)
    scene_list.sort()

    #parameters for view selection
    img_scale = 1.
    max_views = 30
    silhouette_thres = 0.4

    for scene_cnt,scene_name in enumerate(scene_list):

        scene_name = scene_name.rstrip()

        mesh_path = os.path.join(SCANNET_base_path,scene_name,scene_name + '_vh_clean_2.ply')
        meta_file_path = os.path.join(SCANNET_base_path,scene_name,scene_name +'.txt')
        pkl_out_path = os.path.join(SCANNOTATE_PATH,config['out_folder'], scene_name)

        if not os.path.exists(mesh_path):
            continue

        pkl_file = open(os.path.join(pkl_out_path, scene_name + '.pkl'), 'rb')
        scene_obj = pickle.load(pkl_file)
        inst_label_ = scene_obj.inst_seg_3d
        inst_label = np.copy(inst_label_)
        inst_label[inst_label_ == 0] = -255
        inst_label_list_valid = np.unique(inst_label)

        mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)

        # Transfer points to py3d coord system
        T_mat = transform_ScanNet_to_py3D()

        align_mat_Scannet = load_axis_alignment_mat(meta_file_path=meta_file_path)
        align_mat_Scannet = np.reshape(np.asarray(align_mat_Scannet), (4, 4))

        mesh_o3d = alignPclMesh(mesh_o3d, axis_align_matrix=align_mat_Scannet, T=T_mat)

        verts = np.asarray(mesh_o3d.vertices)

        tmesh = Meshes(
            verts=[torch.tensor(verts.astype(np.float32))],
            faces=[torch.tensor(np.asarray(mesh_o3d.triangles))],
        )


        inst_label_norm = inst_label / 255.
        tex = torch.tensor(inst_label_norm.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=-1)
        tex = tex.repeat(1,1,3)
        tmesh.textures = pytorch3d.renderer.mesh.textures.TexturesVertex(verts_features=tex)

        depth_intrinsics_path = os.path.join(SCANNET_base_path, scene_name, str(scene_name) + '.txt')
        _, intrinsics, scene_type = load_intrinsics(depth_intrinsics_path)

        print('Start object mask reprojection')
        _ = view_selection_scannotate(scene_name,inst_label_list_valid,tmesh,SCANNET_base_path,
                                             intrinsics,img_scale,max_views,silhouette_thres,pkl_out_path,scene_obj)
        print('Object mask reprojection done')

if __name__ == "__main__":
    main(parser.parse_args())


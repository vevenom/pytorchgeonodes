import os
import cv2
import numpy as np
import torch
import pickle

from pytorch3d.io import IO

from DataRW.ScanNetRW import ScanNetRW
from ScanNetAnnotation import ScanNetAnnotation, ObjectAnnotation

class ScannotateDatasetMasks(object):
    def __init__(self, scannotate_path):
        self.scannotate_path = scannotate_path

        self.scannet_to_py3d_T = np.zeros((4, 4))
        self.scannet_to_py3d_T[0, 1] = 1
        self.scannet_to_py3d_T[1, 2] = 1
        self.scannet_to_py3d_T[2, 0] = 1
        self.scannet_to_py3d_T[3, 3] = 1

        self.invalid_views_dict = {
            'scene0277_00': {
                '4': ['740', '744']
            },
            'scene0277_01': {
                '1': ['372', '376']
            }
        }

    def yield_obj_frames_dicts(self, scannet_instance: ScanNetRW, scene_name, obj_idx):
        scene_dict = scannet_instance.get_scene_dict(scene_name)

        scene_name = scene_dict['scene_name']

        seg_masks_path = os.path.join(self.scannotate_path, scene_name, 'sam_results_path')

        # name of frames for object start with (obj_idx + 1) followed by '_' and frame number to 6 digits in .png
        # format
        # example: 1_000001.png
        obj_frames = [frame for frame in os.listdir(seg_masks_path) if
                        frame.startswith(str(obj_idx + 1) + '_') and frame.endswith('.png')]
        obj_frames.sort()

        for frame in obj_frames:

            # get frame number and transform it to int
            frame_idx = int(frame.split('_')[1].split('.')[0])
            frame_idx = str(frame_idx)

            frame_dict = scannet_instance.get_frame_dict(scene_dict, frame_idx)

            # update frame_dict with by loading instance segmentation as an image
            frame_dict['instance_seg'] = cv2.imread(os.path.join(seg_masks_path, frame),
                                                    cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) / 255.0
            resize_size = (640 // scannet_instance.downscale_factor, 480 // scannet_instance.downscale_factor)
            frame_dict['instance_seg'] = cv2.resize(frame_dict['instance_seg'],
                                                    resize_size,
                                                    interpolation=cv2.INTER_NEAREST)

            yield frame_dict

    def get_other_objs_pcd(self, scannet_instance, scene_name, obj_idx):
        scene_pcd = scannet_instance.get_scene_pcd(scene_name)
        axis_align_matrix = scannet_instance.get_scene_dict(scene_name)['axisAlignment']
        scene_pcd = np.array(scene_pcd.points)
        scene_pcd_homo = np.ones((scene_pcd.shape[0], 4))
        scene_pcd_homo[:, :3] = scene_pcd
        scene_pcd_homo = scene_pcd_homo.dot(axis_align_matrix.T)
        scene_pcd_homo = scene_pcd_homo.dot(self.scannet_to_py3d_T.T)
        scene_pcd = scene_pcd_homo[:, :3]

        scene_annotation_path = os.path.join(self.scannotate_path, scene_name, scene_name + '.pkl')
        with open(scene_annotation_path, 'rb') as f:
            scene_annotation = pickle.load(f)

        obj_id = scene_annotation.obj_annotation_list[obj_idx].object_id
        other_obj_points = scene_pcd[scene_annotation.inst_seg_3d != obj_id]
        other_obj_points = torch.tensor(other_obj_points, dtype=torch.float32)

        return other_obj_points

    def get_obj_pcd(self, scannet_instance, scene_name, obj_idx):
        scene_pcd = scannet_instance.get_scene_pcd(scene_name)
        axis_align_matrix = scannet_instance.get_scene_dict(scene_name)['axisAlignment']
        scene_pcd = np.array(scene_pcd.points)
        scene_pcd_homo = np.ones((scene_pcd.shape[0], 4))
        scene_pcd_homo[:, :3] = scene_pcd
        scene_pcd_homo = scene_pcd_homo.dot(axis_align_matrix.T)
        scene_pcd_homo = scene_pcd_homo.dot(self.scannet_to_py3d_T.T)
        scene_pcd = scene_pcd_homo[:, :3]

        scene_annotation_path = os.path.join(self.scannotate_path, scene_name, scene_name + '.pkl')
        with open(scene_annotation_path, 'rb') as f:
            scene_annotation = pickle.load(f)

        obj_id = scene_annotation.obj_annotation_list[obj_idx].object_id
        object_points = scene_pcd[scene_annotation.inst_seg_3d == obj_id]
        object_points = torch.tensor(object_points, dtype=torch.float32)

        return object_points

    def get_frame_dict(self, scannet_instance: ScanNetRW, scene_name, obj_idx, frame_id):
        scene_dict = scannet_instance.get_scene_dict(scene_name)

        scene_name = scene_dict['scene_name']

        # seg_masks_path = os.path.join(self.scannotate_path, scene_name, 'mask2d_final')
        seg_masks_path = os.path.join(self.scannotate_path, scene_name, 'sam_results_path')

        scene_annotation_path = os.path.join(self.scannotate_path, scene_name, scene_name + '.pkl')
        with open(scene_annotation_path, 'rb') as f:
            scene_annotation = pickle.load(f)

        frame = str(obj_idx + 1) + '_' + frame_id + '.png'

        # get frame number and transform it to int
        frame_idx = int(frame.split('_')[1].split('.')[0])
        frame_idx = str(frame_idx)

        frame_dict = scannet_instance.get_frame_dict(scene_dict, frame_idx)

        # update frame_dict with by loading instance segmentation as an image
        frame_dict['instance_seg'] = cv2.imread(os.path.join(seg_masks_path, frame),
                                                cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR) / 255.0

        kernel = np.ones((3, 3), np.uint8)
        frame_dict['instance_seg'] = cv2.erode(frame_dict['instance_seg'], kernel)

        resize_size = (640 // scannet_instance.downscale_factor, 480 // scannet_instance.downscale_factor)
        frame_dict['instance_seg'] = cv2.resize(frame_dict['instance_seg'],
                                                resize_size,
                                                interpolation=cv2.INTER_NEAREST)

        return frame_dict


    def generate_obj_batch_dict(self, scannet_instance: ScanNetRW, box_item, scene_name, obj_idx, device):

        batched_frame_dict = {}
        # concatenate all frames in batched_frame_dict

        for frame_ind, frame_id in enumerate(box_item.view_params['frame_ids']):
        # for frame_ind, frame_dict in enumerate(self.yield_obj_frames_dicts(scannet_instance, scene_name, obj_idx)):

            frame_dict = self.get_frame_dict(scannet_instance, scene_name, obj_idx, frame_id)
            frame_name = frame_dict['frame_name']
            # batched_frame_dict[frame_name] = frame_dict

            if scene_name in self.invalid_views_dict.keys():
                if str(obj_idx) in self.invalid_views_dict[scene_name].keys():
                    if frame_name in self.invalid_views_dict[scene_name][str(obj_idx)]:
                        continue

            # color
            # depth
            # pose
            # intrinsics
            # instance_seg
            color = torch.tensor(frame_dict['color'][None], device=device)
            depth = torch.tensor(frame_dict['depth'][None], dtype=torch.float32, device=device)
            pose = torch.tensor(frame_dict['pose'][None], dtype=torch.float32, device=device)
            scannet_to_py3d_T = torch.zeros((4, 4), device=device)
            scannet_to_py3d_T[0, 1] = 1
            scannet_to_py3d_T[1, 2] = 1
            scannet_to_py3d_T[2, 0] = 1
            scannet_to_py3d_T[3, 3] = 1
            pose = torch.matmul(scannet_to_py3d_T, pose)


            view_parameters = box_item.view_params
            R = view_parameters['R'].squeeze(axis=1)[frame_ind][None]
            T = view_parameters['T'].squeeze(axis=1)[frame_ind][None]
            # intrinsics = view_parameters['intrinsics'][frame_ind][None]

            pose_renderer = torch.zeros((R.shape[0], 4, 4), dtype=torch.float32, device=device)
            pose_renderer[:, :3, :3] = torch.tensor(R, dtype=torch.float32, device=device)
            pose_renderer[:, :3, 3] = torch.tensor(T, dtype=torch.float32, device=device)
            pose_renderer[:, 3, 3] = 1

            intrinsics = torch.tensor(frame_dict['intrinsics'][None], device=device)
            instance_seg = torch.tensor(frame_dict['instance_seg'][None], device=device)
            if len(instance_seg.shape) == 4:
                instance_seg = instance_seg[..., 0]

            if 'color' in batched_frame_dict:
                batched_frame_dict['color'] = torch.cat([batched_frame_dict['color'], color], dim=0)
                batched_frame_dict['depth'] = torch.cat([batched_frame_dict['depth'], depth], dim=0)
                batched_frame_dict['pose'] = torch.cat([batched_frame_dict['pose'], pose], dim=0)
                batched_frame_dict['pose_renderer'] = torch.cat([batched_frame_dict['pose_renderer'],
                                                                 pose_renderer], dim=0)
                batched_frame_dict['intrinsics'] = (
                    torch.cat([batched_frame_dict['intrinsics'], intrinsics], dim=0))
                batched_frame_dict['instance_seg'] = (
                    torch.cat([batched_frame_dict['instance_seg'], instance_seg], dim=0))

                batched_frame_dict['frame_name'].append(frame_name)
            else:
                batched_frame_dict['color'] = color
                batched_frame_dict['depth'] = depth
                batched_frame_dict['pose'] = pose
                batched_frame_dict['pose_renderer'] = pose_renderer
                batched_frame_dict['intrinsics'] = intrinsics
                batched_frame_dict['instance_seg'] = instance_seg
                batched_frame_dict['frame_name'] = [frame_name]


        obj_pcd = self.get_obj_pcd(scannet_instance, scene_name, obj_idx)
        other_obj_pcd = self.get_other_objs_pcd(scannet_instance, scene_name, obj_idx)
        batched_frame_dict['object_points'] = obj_pcd.to(device)
        batched_frame_dict['other_object_points'] = other_obj_pcd.to(device)

        return batched_frame_dict

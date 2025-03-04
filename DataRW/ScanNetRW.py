import os
import numpy as np
import cv2
import open3d as o3d
import torch

class ScanNetRW(object):
    """
    Reading and writing ScanNet data
    """
    def __init__(self, scannet_processed_path, scannet_ply_path, use_alternative_names):
        self.scannet_path = scannet_processed_path
        self.scannet_ply_path = scannet_ply_path

        self.scene_list = self.get_scenes_list()

        self.downscale_factor = 1 # keep at 1, otherwise fix mesh scaling for Scan2Cad or Scannotate

        self.alternative_naming = use_alternative_names  # frame-******.color.jpg

    def get_scene_ply_path(self, scene):
        scene_ply_path = os.path.join(self.scannet_ply_path, scene, scene + '_vh_clean_2.ply')
        return scene_ply_path

    def get_scene_path(self, scene):
        scene_path = os.path.join(self.scannet_path, scene)
        return scene_path

    def get_scenes_list(self):
        scene_list = os.listdir(self.scannet_path)
        scene_list = [scene for scene in scene_list if scene[0] != '.']
        scene_list = [scene for scene in scene_list if scene[:5] == 'scene']
        scene_list.sort()

        return scene_list

    def get_scene_dict(self, scene_name):
        scene_dict = {}
        scene_dict['scene_path'] = self.get_scene_path(scene_name)
        scene_dict['scene_name'] = scene_name
        scene_dict['scene_ply_path'] = self.get_scene_ply_path(scene_name)

        color_postfix = '.jpg'
        color_folder_name = 'color'
        prefix = ''
        if self.alternative_naming:
            color_postfix = '.color.jpg'
            color_folder_name = 'images'
            prefix = 'frame-'

        color_imgs_path = os.path.join(scene_dict['scene_path'], color_folder_name)
        color_imgs_list = os.listdir(color_imgs_path)
        color_imgs_list = [str(int(img[len(prefix):-len(color_postfix)])) for img in color_imgs_list if img[0] != '.']

        scene_dict['frame_names'] = color_imgs_list
        scene_dict['frame_names'].sort(key=lambda x: int(x))

        if not self.alternative_naming:
            intrinsics_path = os.path.join(scene_dict['scene_path'], 'intrinsic', 'intrinsic_depth.txt')
            intrinsics = np.loadtxt(intrinsics_path)
        else:
            #577.590698 0.000000 318.905426 0.000000
            # 0.000000 578.729797 242.683609 0.000000
            # 0.000000 0.000000 1.000000 0.000000
            # 0.000000 0.000000 0.000000 1.000000
            intrinsics = np.array([[577.590698, 0.000000, 318.905426, 0.000000],
                                   [0.000000, 578.729797, 242.683609, 0.000000],
                                   [0.000000, 0.000000, 1.000000, 0.000000],
                                      [0.000000, 0.000000, 0.000000, 1.000000]])

        intrinsics[0, 0] /= self.downscale_factor
        intrinsics[1, 1] /= self.downscale_factor
        intrinsics[0, 2] /= self.downscale_factor
        intrinsics[1, 2] /= self.downscale_factor

        scene_dict['intrinsics'] = intrinsics

        axis_alignment_path = os.path.join(scene_dict['scene_path'], scene_name + '.txt')
        # axisAlignment = 0.945519 0.325568 0.000000 -5.384390 -0.325568 0.945519 0.000000 -2.871780 0.000000 0.000000 1.000000 -0.064350 0.000000 0.000000 0.000000 1.000000
        # colorHeight = 968
        # colorToDepthExtrinsics = 0.999973 0.006791 0.002776 -0.037886 -0.006767 0.999942 -0.008366 -0.003410 -0.002833 0.008347 0.999961 -0.021924 -0.000000 0.000000 -0.000000 1.000000
        # colorWidth = 1296
        # depthHeight = 480
        # depthWidth = 640
        # fx_color = 1170.187988
        # fx_depth = 571.623718
        # fy_color = 1170.187988
        # fy_depth = 571.623718
        # mx_color = 647.750000
        # mx_depth = 319.500000
        # my_color = 483.750000
        # my_depth = 239.500000
        # numColorFrames = 5578
        # numDepthFrames = 5578
        # numIMUmeasurements = 11834
        # sceneType = Apartment

        with open(axis_alignment_path, 'r') as f:
            scene_params = f.read()
        scene_params = scene_params.split('\n')
        scene_params = [param for param in scene_params if param != '']
        scene_params = [param.split(' = ') for param in scene_params]
        scene_params = {param[0]: param[1] for param in scene_params}

        # save axis_alignment as np matrix in one line
        if 'axisAlignment' in scene_params:
            axis_alignment = np.fromstring(scene_params['axisAlignment'], sep=' ')
            scene_dict['axisAlignment'] = axis_alignment.reshape((4, 4))
        else:
            raise Exception('axisAlignment not found in scene_params')

        return scene_dict

    def get_scene_pcd(self, scene_name):
        scene_dict = self.get_scene_dict(scene_name)
        scene_pcd = o3d.io.read_point_cloud(scene_dict['scene_ply_path'])

        return scene_pcd

    def scannet_pose_to_py3d(self, pose, use_torch=False):
        # Transform to P3D coordinate system
        T_py3d = np.eye(4)
        T_py3d[0, 0] = -1
        T_py3d[1, 1] = -1
        if use_torch:
            T_py3d = torch.from_numpy(T_py3d).to(pose.device, torch.float32)
            pose = T_py3d.matmul(pose)
        else:
            pose = T_py3d.dot(pose)

        return pose

    def scannet_pose_to_batched_py3d(self, pose_batch):

        T_py3d = np.eye(4)
        T_py3d[0, 0] = -1
        T_py3d[1, 1] = -1

        T_py3d = torch.from_numpy(T_py3d).float().to(pose_batch.device)[None]
        T_py3d = T_py3d.expand(pose_batch.shape[0], -1, -1)

        pose_batch_transf = torch.bmm(T_py3d, pose_batch)

        return pose_batch_transf

    def yield_scene_frames_dicts(self, scene_name):
        scene_dict = self.get_scene_dict(scene_name)
        scene_path = scene_dict['scene_path']
        frames = scene_dict['frame_names']

        for frame in frames:
            frame_dict = self.get_frame_dict(scene_dict, frame)
            yield frame_dict

    def get_frame_dict(self, scene_dict, frame_name):

        color_postfix = '.jpg'
        depth_postfix = '.png'
        color_folder_name = 'color'
        depth_folder_name = 'depth'
        if self.alternative_naming:
            frame_name = 'frame-' + format(int(frame_name), '06d')
            color_postfix = '.color.jpg'
            depth_postfix = '.depth.pgm'
            color_folder_name = 'images'
            depth_folder_name = 'depths'

        scene_path = scene_dict['scene_path']
        frame_dict = {}
        frame_dict['scene_path'] = scene_path
        scene_dict['scene_name'] = scene_dict['scene_name']
        frame_dict['frame_name'] = frame_name

        resize_size = (640 // self.downscale_factor, 480 // self.downscale_factor)

        frame_dict['color'] = cv2.imread(os.path.join(scene_path, color_folder_name, frame_name + color_postfix))
        frame_dict['color'] = cv2.resize(frame_dict['color'],
                                         resize_size)
        frame_dict['color'][..., [0,1,2]] = frame_dict['color'][..., [2,1,0]]

        frame_dict['depth'] = (
            cv2.imread(os.path.join(scene_path, depth_folder_name, frame_name + depth_postfix),
                        cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)) / 1000.0
        frame_dict['depth'] = cv2.resize(frame_dict['depth'],
                                         resize_size,
                                         interpolation=cv2.INTER_NEAREST)
        frame_dict['pose'] = np.loadtxt(os.path.join(scene_path, 'pose', frame_name + '.txt'))
        frame_dict['pose'] = np.matmul(scene_dict['axisAlignment'], frame_dict['pose'])

        # frame_dict['pose'] = np.linalg.inv(frame_dict['pose'])

        frame_dict['intrinsics'] = scene_dict['intrinsics']
        return frame_dict


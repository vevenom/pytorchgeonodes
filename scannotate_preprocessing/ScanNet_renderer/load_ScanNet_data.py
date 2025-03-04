import numpy as np
import os
from utils import transform_ScanNet_to_py3D,Ry,Rx,Rz

def load_intrinsics(depth_intrinsics_path):

    cam_K = np.zeros((3, 3))
    cam_K[2, 2] = 1
    cam_K_depth = np.zeros_like(cam_K)
    scene_type = None
    with open(depth_intrinsics_path, 'r') as cam_params:
        for cam_param in cam_params:
            param_split = cam_param.split(' ')
            if param_split[0] == 'fx_color':
                cam_K[0, 0] = float(param_split[2])
            if param_split[0] == 'fy_color':
                cam_K[1, 1] = float(param_split[2])
            if param_split[0] == 'mx_color':
                cam_K[0, 2] = float(param_split[2])
            if param_split[0] == 'my_color':
                cam_K[1, 2] = float(param_split[2])

            if param_split[0] == 'fx_depth':
                cam_K_depth[0, 0] = float(param_split[2])
            if param_split[0] == 'fy_depth':
                cam_K_depth[1, 1] = float(param_split[2])
            if param_split[0] == 'mx_depth':
                cam_K_depth[0, 2] = float(param_split[2])
            if param_split[0] == 'my_depth':
                cam_K_depth[1, 2] = float(param_split[2])

            if param_split[0] == 'sceneType':
                #param_test = 'sceneType = Living room / Lounge'
                #param_test = 'sceneType = Kitchen'
                #param_split = param_test.split(' ')
                scene_type = ' '.join(param_split[2:]).rstrip()

    return cam_K,cam_K_depth,scene_type

def load_axis_alignment_mat(meta_file_path):
    metaFile = meta_file_path  # includes axisAlignment info for the train set scans.
    assert os.path.exists(metaFile), '%s' % metaFile
    axis_align_matrix = np.identity(4)
    if os.path.isfile(metaFile):
        with open(metaFile) as f:
            lines = f.readlines()

        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) \
                                     for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break

    return axis_align_matrix

# TODO delete; code depricted
# def get_pose_at_idx(meta_file_path,trans_path, pose_path,idx, use_axis_aligned=True):
#     pose_list = []
#     pose_path = os.path.join(pose_path, str(idx) + '.txt')
#     if not os.path.exists(pose_path):
#         return None
#     pose = np.loadtxt(pose_path)
#     if np.any(np.isnan(pose)):
#         return None
#     if np.any(np.isinf(pose)):
#         return None
#     if use_axis_aligned:
#         #if os.path.exists(trans_path):
#         #    addTrans = np.load(trans_path)
#         axis_align_matrix = np.asarray(load_axis_alignment_mat(meta_file_path)).reshape((4, 4))
#         #    axis_align_matrix = addTrans.dot(axis_align_matrix)
#     else:
#         axis_align_matrix = np.eye(4)
#
#     T = transform_ScanNet_to_py3D()
#
#     # M_frame_inv is world to cam
#     M_frame_inv = np.linalg.inv(pose)
#     M_frame_inv = np.dot(M_frame_inv, np.linalg.inv(axis_align_matrix))  # for mcss_output
#     M_frame_inv = np.dot(M_frame_inv, np.linalg.inv(T))
#     rot_tmp3 = Rz(np.deg2rad(180))
#     T2 = np.eye(4)
#     T2[:3, :3] = rot_tmp3
#     M_frame_inv = np.dot(T2, M_frame_inv)
#
#     rot_py3d = np.copy(M_frame_inv[:3, :3].T)
#     M_frame_inv[:3, :3] = rot_py3d
#     pose_list.append(M_frame_inv)
#
#     return pose_list


def get_pose_at_idx_scannet25k(meta_file_path, pose_path, idx, use_axis_aligned=True):
    pose_list = []
    #pose_path = os.path.join(pose_path, 'frame-' + str(idx).zfill(6) + '.pose.txt')
    pose_path = os.path.join(pose_path, str(idx) + '.txt')

    if not os.path.exists(pose_path):
        return None
    pose = np.loadtxt(pose_path)
    if np.any(np.isnan(pose)):
        return None
    if np.any(np.isinf(pose)):
        return None
    if use_axis_aligned:
        axis_align_matrix = np.asarray(load_axis_alignment_mat(meta_file_path)).reshape((4, 4))
    else:
        axis_align_matrix = np.eye(4)

    T = transform_ScanNet_to_py3D()

    M_frame_inv = np.linalg.inv(pose)
    M_frame_inv = np.dot(M_frame_inv, np.linalg.inv(axis_align_matrix))
    M_frame_inv = np.dot(M_frame_inv, np.linalg.inv(T))
    T2 = np.eye(4)
    T2[:3, :3] = Rz(np.deg2rad(180))
    M_frame_inv = np.dot(T2, M_frame_inv)

    rot_py3d = np.copy(M_frame_inv[:3, :3].T)
    M_frame_inv[:3, :3] = rot_py3d
    pose_list.append(M_frame_inv)

    return pose_list


def get_pose_at_idx(meta_file_path, pose_path, idx, use_axis_aligned=True):
    pose_list = []
    pose_path = os.path.join(pose_path, str(idx) + '.txt')
    if not os.path.exists(pose_path):
        return None
    pose = np.loadtxt(pose_path)
    if np.any(np.isnan(pose)):
        return None
    if np.any(np.isinf(pose)):
        return None
    if use_axis_aligned:
        axis_align_matrix = np.asarray(load_axis_alignment_mat(meta_file_path)).reshape((4, 4))
    else:
        axis_align_matrix = np.eye(4)

    T = transform_ScanNet_to_py3D()

    M_frame_inv = np.linalg.inv(pose)
    M_frame_inv = np.dot(M_frame_inv, np.linalg.inv(axis_align_matrix))
    M_frame_inv = np.dot(M_frame_inv, np.linalg.inv(T))
    T2 = np.eye(4)
    T2[:3, :3] = Rz(np.deg2rad(180))
    M_frame_inv = np.dot(T2, M_frame_inv)

    rot_py3d = np.copy(M_frame_inv[:3, :3].T)
    M_frame_inv[:3, :3] = rot_py3d
    pose_list.append(M_frame_inv)

    return pose_list

def get_pose_at_idx_video_rendering(meta_file_path, pose_path, idx, use_axis_aligned=True):
    pose_list = []
    pose_path = os.path.join(pose_path, str(idx) + '.pose.txt')
    if not os.path.exists(pose_path):
        return None
    pose = np.loadtxt(pose_path)
    if np.any(np.isnan(pose)):
        return None
    if np.any(np.isinf(pose)):
        return None
    if use_axis_aligned:
        axis_align_matrix = np.asarray(load_axis_alignment_mat(meta_file_path)).reshape((4, 4))
    else:
        axis_align_matrix = np.eye(4)

    T = transform_ScanNet_to_py3D()

    M_frame_inv = np.linalg.inv(pose)
    M_frame_inv = np.dot(M_frame_inv, np.linalg.inv(axis_align_matrix))
    M_frame_inv = np.dot(M_frame_inv, np.linalg.inv(T))
    T2 = np.eye(4)
    T2[:3, :3] = Rz(np.deg2rad(180))
    M_frame_inv = np.dot(T2, M_frame_inv)

    rot_py3d = np.copy(M_frame_inv[:3, :3].T)
    M_frame_inv[:3, :3] = rot_py3d
    pose_list.append(M_frame_inv)

    return pose_list

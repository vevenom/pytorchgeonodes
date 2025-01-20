from typing import Union
import torch
import numpy as np
import cv2
import copy
import quaternion
import os
from torch.cuda import device

from pytorch3d.loss.chamfer import _validate_chamfer_reduction_inputs, _handle_pointcloud_input
from pytorch3d.ops.knn import knn_points
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import Translate
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesVertex


def parse_params_from_json(json_params, device):
    """
    Parse parameters from json file
    """

    from pytorch3d.transforms import axis_angle_to_matrix

    # Load parameters from json file
    input_params_dict = {}
    angle_x_rad = None
    for key, value in json_params['input_dict'].items():
        if key not in ['OBJ_Rotation', 'translation_offset']:
            input_params_dict[key] = torch.tensor([[value]], device=device)
    angle_x_rad = torch.tensor([[json_params['rotation_angle_y']]], device=device)
    translation_offset = torch.tensor(json_params['translation_offset'], device=device)

    # angle_x_degree = angle_x_rad * 180.0 / np.pi
    angle_tensor = torch.zeros_like(angle_x_rad)
    angle_tensor = angle_tensor.repeat(*angle_x_rad.shape[:-1], 3)
    angle_tensor[:, 1] = angle_x_rad

    # create rotation matrix
    # create eye matrix
    rot_matrix = torch.eye(4, device=angle_tensor.device)
    rot_matrix = rot_matrix[None].repeat(*angle_tensor.shape[:-1], 1, 1)
    rot_matrix[:, :3, :3] = axis_angle_to_matrix(angle_tensor)

    return input_params_dict, rot_matrix, translation_offset


def calculate_mesh_from_input_dict(self, input_params_dict, rotation_matrix, translation_offset=None):

    from pytorch3d.transforms import Transform3d, Translate
    from pytorch3d.structures import Meshes

    device = self.device

    #         start.record()
    _, outputs = self.geometry_nodes.forward(input_params_dict, transform2blender_coords=True)
    obj_mesh = outputs[0][0][0]
    obj_mesh = Meshes(verts=obj_mesh.verts, faces=obj_mesh.faces)

    #         end.record()

    #         torch.cuda.synchronize()

    # print('Time needed for geometry nodes: {}'.format(start.elapsed_time(end)))

    verts = obj_mesh.verts_packed()
    faces = obj_mesh.faces_packed()

    # To Scan2CAD
    # T_scan2CAD = np.eye(4)

    # T_scan2CAD = np.array([[0, 1, 0, 0],
    #                        [0, 0, 1, 0],
    #                        [-1, 0, 0, 0],
    #                        [0, 0, 0, 1]])
    # # T_scan2CAD = np.array([[0, 1, 0, 0],
    # #                        [-1, 0, 0, 0],
    # #                        [0, 0, 1, 0],
    # #                        [0, 0, 0, 1]])
    # T_scan2CAD = torch.from_numpy(T_scan2CAD).float().to(device)

    transform = Transform3d(device=device)

    if rotation_matrix is not None:
        transform_rot = Transform3d(matrix=rotation_matrix, device=device)
        assert rotation_matrix.shape[0] == 1
        transform = transform.compose(transform_rot)
    else:
        assert False, "We are doing experiments with rotations"

    # transform = transform.compose(Transform3d(matrix=T_scan2CAD))

    # If object is not at origin
    bb = obj_mesh.get_bounding_boxes()  # (N, 3, 2)

    bb_center = (bb[:, :, 1] - bb[:, :, 0]) / 2 + bb[:, :, 0]

    verts = verts - bb_center

    mesh = Meshes(verts=[verts], faces=[faces], textures=obj_mesh.textures)
    bb = mesh.get_bounding_boxes()  # (N, 3, 2)

    bb_center = (bb[:, :, 1] - bb[:, :, 0]) / 2 + bb[:, :, 0]

    verts = transform.transform_points(verts)

    obj_mesh = Meshes(verts=[verts], faces=[faces], textures=obj_mesh.textures)

    bb_center = transform.transform_points(bb_center)

    translation = self.obj_center + bb_center
    verts = verts + translation

    # # # if object is centered at origin
    # verts = transform.transform_points(verts)
    # verts = verts + self.obj_center

    if translation_offset is not None:
        translate = Translate(translation_offset)
        verts = translate.transform_points(verts)

    obj_mesh = Meshes(verts=[verts], faces=[faces], textures=obj_mesh.textures)

    return obj_mesh


def load_and_transform_mesh(model_path, cad_transform,device):
    from pytorch3d.io import load_obj
    from misc.utils_CAD_retrieval import normalize_mesh

    verts, faces, _ = load_obj(model_path, load_textures=False, device=device)

    mesh, tverts_normalized = normalize_mesh(verts, faces.verts_idx,device)

    bb = mesh.get_bounding_boxes()  # (N, 3, 2)

    # bb_center = (bb[:, :, 1] - bb[:, :, 0]) / 2 + bb[:, :, 0]
    #
    # print(bb_center)
    # assert False

    # transform normalized mesh to final position using transformaiton matrix
    tverts = cad_transform.transform_points(tverts_normalized)

    cad_mesh = Meshes(
        verts=[tverts.squeeze(dim=0)],
        faces=[faces.verts_idx],
    )

    return cad_mesh

def read_object_pose_from_annotation(box_item):

    # To extract transformation matrix M use get_matrix(), but it is in pytorch3d format, see below:
    # M = [
    #     [Rxx, Ryx, Rzx, 0],
    #     [Rxy, Ryy, Rzy, 0],
    #     [Rxz, Ryz, Rzz, 0],
    #     [Tx, Ty, Tz, 1],
    # ]

    cad_transformation = box_item.transform3d.to('cpu')  # type:pytorch3d.transforms.transform3d.Transform3d

    cad_transformation = cad_transformation.get_matrix().detach().numpy()[0].T

    # Object translation, e.g. center of object
    obj_translation = cad_transformation[:3, 3]

    # Object rotation as matrix and quaterions
    obj_rot_matrix = cad_transformation[:3, :3]
    obj_rotation = quaternion.from_rotation_matrix(obj_rot_matrix)
    obj_rotation = quaternion.as_float_array(obj_rotation)

    # scale in x,y,z directions
    sx = np.linalg.norm(obj_rot_matrix[0:3, 0])
    sy = np.linalg.norm(obj_rot_matrix[0:3, 1])
    sz = np.linalg.norm(obj_rot_matrix[0:3, 2])

    obj_scale = np.array([sx, sy, sz])

    obj_pose_dict= {
        'center': obj_translation.tolist(),
        'rotation': obj_rotation.tolist(),
        'scale': obj_scale.tolist()
    }

    return obj_pose_dict

def get_object_center(box_item, return_scale=True):

    obj_pose_dict = read_object_pose_from_annotation(box_item)

    obj_rotation = quaternion.as_rotation_matrix(np.quaternion(*obj_pose_dict['rotation']))
    # obj_rotation = np.linalg.inv(obj_rotation)

    obj_center = np.array(obj_pose_dict['center'])[:, None]

    # obj_center = np.matmul(obj_rotation, obj_center)

    obj_center = torch.from_numpy(obj_center).float()
    obj_center = obj_center.permute(1, 0)

    if return_scale:
        obj_scale = np.array(obj_pose_dict['scale'])[None, :]
        obj_scale = torch.from_numpy(obj_scale)

        return obj_center, obj_scale

    return obj_center


# def get_object_bottom(box_item):
#
#     obj_pose_dict = read_object_pose_from_annotation(box_item)
#
#     obj_rotation = quaternion.as_rotation_matrix(np.quaternion(*obj_pose_dict['rotation']))
#     # obj_rotation = np.linalg.inv(obj_rotation)
#
#     obj_center = np.array(obj_pose_dict['center'])[:, None]
#     obj_scale = np.array(obj_pose_dict['scale'])[:, None]
#
#     # obj_center = np.matmul(obj_rotation, obj_center)
#
#     obj_center = torch.from_numpy(obj_center).float()
#     obj_center = obj_center.permute(1, 0)
#
#     obj_bottom = obj_center
#     print(obj_center.shape)
#     print(obj_scale.shape)
#     assert False
#
#     return obj_center

def normalize_mesh(verts_init, faces, device):
    mesh = Meshes(
        verts=[verts_init],
        faces=[faces],
    )


    bbox = mesh.get_bounding_boxes().squeeze(dim=0)
    bbox = bbox.cpu().detach().numpy()

    center = torch.tensor(bbox.mean(1)).float().to(device)
    vector_x = np.array([bbox[0, 1] - bbox[0, 0], 0, 0])
    vector_y = np.array([0, bbox[1, 1] - bbox[1, 0], 0])
    vector_z = np.array([0, 0, bbox[2, 1] - bbox[2, 0]])

    coeff_x = np.linalg.norm(vector_x)
    coeff_y = np.linalg.norm(vector_y)
    coeff_z = np.linalg.norm(vector_z)

    mesh = mesh.offset_verts(-center)
    transform_func = Transform3d().scale(x=(1 / coeff_x), y=(1 / coeff_y), z=1 / coeff_z).to(device)
    tverts = transform_func.transform_points(mesh.verts_list()[0]).unsqueeze(dim=0)

    mesh = Meshes(
        verts=[tverts.squeeze(dim=0)],
        faces=[faces]
    )

    return mesh, tverts


def two_way_chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    clamp_max=0.2,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
    norm: int = 2,
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    # return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    # clamp to avoid wrong correspondence
    cham_x = torch.clamp_max(cham_x, max=clamp_max)
    cham_y = torch.clamp_max(cham_y, max=clamp_max)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if point_reduction == "mean":
        x_lengths_clamped = x_lengths.clamp(min=1)
        y_lengths_clamped = y_lengths.clamp(min=1)
        cham_x /= x_lengths_clamped
        cham_y /= y_lengths_clamped

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else max(N, 1)
            cham_x /= div
            cham_y /= div

    cham_dist =  cham_x + cham_y

    return cham_dist


def one_way_chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    clamp_max=0.2,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
    norm: int = 2,
):
    """
    Chamfer distance between two pointclouds x and y.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
        norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
          :param clamp_max:
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]


    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)

    # clamp to avoid wrong correspondence
    cham_x = torch.clamp_max(cham_x, max=clamp_max)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0


    if weights is not None:
        cham_x *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    # cham_y = cham_y.sum(1)  # (N,)

    if point_reduction == "mean":
        x_lengths_clamped = x_lengths.clamp(min=1)
        cham_x /= x_lengths_clamped

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else max(N, 1)
            cham_x /= div

    cham_dist = cham_x
    return cham_dist


def calculate_floor_plane(object_center, scale):
    scale_half = scale / 2

    p1 = torch.tensor([object_center[:, 0] - scale_half[:, 0],
                       object_center[:, 1] - scale_half[:, 1],
                       object_center[:, 2] - scale_half[:, 2]]).to(object_center.device)

    p2 = torch.tensor([object_center[:, 0] + scale_half[:, 0],
                       object_center[:, 1] - scale_half[:, 1],
                       object_center[:, 2] - scale_half[:, 2]]).to(object_center.device)

    p3 = torch.tensor([object_center[:, 0] - scale_half[:, 0],
                       object_center[:, 1] - scale_half[:, 1],
                       object_center[:, 2] + scale_half[:, 2]]).to(object_center.device)
    # p4 = torch.tensor([object_center[:, 0] + scale_half[:, 0],
    #                     object_center[:, 1] - scale_half[:, 1],
    #                     object_center[:, 2] + scale_half[:, 2]]).to(object_center.device)

    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = torch.cross(v1, v2)
    cp = cp / torch.linalg.norm(cp, keepdim=True)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = -torch.dot(cp, p3)

    floor_plane = torch.tensor([a,b,c,d], device=object_center.device)

    return floor_plane
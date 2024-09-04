# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import Rotate, Transform3d, Translate
from pytorch3d.renderer.cameras import CamerasBase, _get_sfm_calibration_matrix
from pytorch3d.structures import Pointclouds


# Default values for rotation and translation matrices.
r = np.expand_dims(np.eye(3), axis=0)  # (1, 3, 3)
t = np.expand_dims(np.zeros(3), axis=0)  # (1, 3)

class SfMPerspectiveCamerasScanNet(CamerasBase):
    """
    A class which stores a batch of parameters to generate a batch of
    transformation matrices using the multi-view geometry convention for
    perspective camera.
    """

    def __init__(
            self, focal_length=1.0, principal_point=((0.0, 0.0),), R=r, T=t, device="cpu"
    ):
        """
        __init__(self, focal_length, principal_point, R, T, device) -> None

        Args:
            focal_length: Focal length of the camera in world units.
                A tensor of shape (N, 1) or (N, 2) for
                square and non-square pixels respectively.
            principal_point: xy coordinates of the center of
                the principal point of the camera in pixels.
                A tensor of shape (N, 2).
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            device: torch.device or string
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        # R = torch.transpose(R, 1, 2)

        super().__init__(
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
        )

    def get_camera_center(self, **kwargs) -> torch.Tensor:
        """
        Return the 3D location of the camera optical center
        in the world coordinates.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting T here will update the values set in init as this
        value may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            C: a batch of 3D locations of shape (N, 3) denoting
            the locations of the center of each camera in the batch.
        """
        w2v_trans = self.get_world_to_view_transform(**kwargs)
        P = w2v_trans.inverse().get_matrix()
        # the camera center is the translation component (the first 3 elements
        # of the last row) of the inverted world-to-view
        # transform (4x4 RT matrix)
        C = P[:, 3, :3]
        # C = transform_scannet_points(C)
        # C[..., 0] /= 320
        # C[..., 1] /= 320
        print("C", C)
        return C

    def get_world_to_view_transform(self, **kwargs) -> Transform3d:
        """
        Return the world-to-view transform.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting R and T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            T: a Transform3d object which represents a batch of transforms
            of shape (N, 3, 3)
        """
        self.R = kwargs.get("R", self.R)  # pyre-ignore[16]
        self.T = kwargs.get("T", self.T)  # pyre-ignore[16]
        world_to_view_transform = get_world_to_view_transform(R=self.R, T=self.T)
        print("RT", world_to_view_transform.get_matrix())
        return world_to_view_transform

    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the projection matrix using the
        multi-view geometry convention.

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in __init__.

        Returns:
            P: A `Transform3d` object with a batch of `N` projection transforms.

        .. code-block:: python

            fx = focal_length[:, 0]
            fy = focal_length[:, 1]
            px = principal_point[:, 0]
            py = principal_point[:, 1]

            P = [
                    [fx,   0,   px,   0],
                    [0,   fy,   py,   0],
                    [0,    0,    0,   1],
                    [0,    0,    1,   0],
            ]
        """
        # pyre-ignore[16]
        principal_point = kwargs.get("principal_point", self.principal_point)
        # pyre-ignore[16]
        focal_length = kwargs.get("focal_length", self.focal_length)

        # P = _get_sfm_calibration_matrix(
        #     self._N, self.device, focal_length, principal_point, False
        # )

        P = _get_sfm_calibration_matrix_scannet(
                    self._N, self.device, focal_length, principal_point, False
                )


        transform = Transform3d(device=self.device)
        transform._matrix = P.transpose(1, 2).contiguous()

        print("P", transform._matrix)

        return transform

    def unproject_points(
            self, xy_depth: torch.Tensor, world_coordinates: bool = True, **kwargs
    ) -> torch.Tensor:
        assert False
        if world_coordinates:
            to_screen_transform = self.get_full_projection_transform(**kwargs)
        else:
            to_screen_transform = self.get_projection_transform(**kwargs)

        unprojection_transform = to_screen_transform.inverse()
        xy_inv_depth = torch.cat(
            (xy_depth[..., :2], 1.0 / xy_depth[..., 2:3]), dim=-1  # type: ignore
        )
        return unprojection_transform.transform_points(xy_inv_depth)

    def transform_points(
            self, points, eps: Optional[float] = None, **kwargs
    ) -> torch.Tensor:
        """
        Transform input points from world to screen space.

        Args:
            points: torch tensor of shape (..., 3).
            eps: If eps!=None, the argument is used to clamp the
                divisor in the homogeneous normalization of the points
                transformed to the screen space. Plese see
                `transforms.Transform3D.transform_points` for details.

                For `CamerasBase.transform_points`, setting `eps > 0`
                stabilizes gradients since it leads to avoiding division
                by excessivelly low numbers for points close to the
                camera plane.

        Returns
            new_points: transformed points with the same shape as the input.
        """
        print("transform_points?")
        world_to_view_transform = self.get_world_to_view_transform(R=self.R, T=self.T)
        view_to_screen_transform = self.get_projection_transform(**kwargs)

        # ndc_view = world_to_view_transform.transform_points(points, eps=eps)
        # ndc_view = view_to_screen_transform.transform_points(screen_view, eps=eps)
        # screen_view = ndc_view
        # screen_points = transform_scannet_points(screen_view)

        # view_points = world_to_view_transform.transform_points(points, eps=eps)
        # view_points = transform_scannet_points(view_points)
        # print("view_points", view_points)
        # screen_points = view_to_screen_transform.transform_points(view_points, eps=eps)
        # print("screen_points", screen_points)

        # world_to_screen_transform = self.get_full_projection_transform(**kwargs)
        # ndc_view = world_to_screen_transform.transform_points(points, eps=eps)
        # screen_points = transform_scannet_points(ndc_view)

        # pytorch3d_points = transform_scannet_points(points)
        world_to_screen_transform = self.get_full_projection_transform(**kwargs)
        screen_points = ndc_view = world_to_screen_transform.transform_points(points, eps=eps)

        # print("screen_points", screen_points)
        print("world_to_screen_transform", world_to_screen_transform)

        return screen_points

def transform_scannet_points(points):
    transformed_points = torch.zeros_like(points)
    transformed_points[..., 0] = -1 * points[..., 0] #/ 320
    transformed_points[..., 1] = -1 * points[..., 1] #/ 320
    transformed_points[..., 2] = 1 * points[..., 2]
    # screen_view[:, :, 2] *= -1

    return transformed_points


# SfMCameras helper
def _get_sfm_calibration_matrix_scannet(
        N, device, focal_length, principal_point, orthographic: bool
) -> torch.Tensor:
    """
    Returns a calibration matrix of a perspective/orthograpic camera.

    Args:
        N: Number of cameras.
        focal_length: Focal length of the camera in world units.
        principal_point: xy coordinates of the center of
            the principal point of the camera in pixels.

        The calibration matrix `K` is set up as follows:

        .. code-block:: python

            fx = focal_length[:,0]
            fy = focal_length[:,1]
            px = principal_point[:,0]
            py = principal_point[:,1]

            for orthographic==True:
                K = [
                        [fx,   0,    0,  px],
                        [0,   fy,    0,  py],
                        [0,    0,    1,   0],
                        [0,    0,    0,   1],
                ]
            else:
                K = [
                        [fx,   0,   px,   0],
                        [0,   fy,   py,   0],
                        [0,    0,    0,   1],
                        [0,    0,    1,   0],
                ]

    Returns:
        A calibration matrix `K` of the SfM-conventioned camera
        of shape (N, 4, 4).
    """

    if not torch.is_tensor(focal_length):
        focal_length = torch.tensor(focal_length, device=device)

    if len(focal_length.shape) in (0, 1) or focal_length.shape[1] == 1:
        fx = fy = focal_length
    else:
        fx, fy = focal_length.unbind(1)

    if not torch.is_tensor(principal_point):
        principal_point = torch.tensor(principal_point, device=device)

    px, py = principal_point.unbind(1)

    K = fx.new_zeros(N, 4, 4)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    if orthographic:
        # K[:, 0, 3] = px
        # K[:, 1, 3] = py
        # K[:, 2, 2] = 1.0
        # K[:, 3, 3] = 1.0
        assert False, "Not implemented properly"
    else:
        K[:, 0, 2] = px
        K[:, 1, 2] = py

        K[:, 3, 2] = 1.0
        K[:, 2, 3] = 1.0

        # K[:, 2, 2] = 1.0
        # K[:, 3, 3] = 1.0

    print("K", K)

    return K

################################################
# Helper functions for world to view transforms
################################################

def get_world_to_view_transform(R=r, T=t) -> Transform3d:
    """
    This function returns a Transform3d representing the transformation
    matrix to go from world space to view space by applying a rotation and
    a translation.

    PyTorch3D uses the same convention as Hartley & Zisserman.
    I.e., for camera extrinsic parameters R (rotation) and T (translation),
    we map a 3D point `X_world` in world coordinates to
    a point `X_cam` in camera coordinates with:
    `X_cam = X_world R + T`

    Args:
        R: (N, 3, 3) matrix representing the rotation.
        T: (N, 3) matrix representing the translation.

    Returns:
        a Transform3d object which represents the composed RT transformation.

    """
    # TODO: also support the case where RT is specified as one matrix
    # of shape (N, 4, 4).

    if T.shape[0] != R.shape[0]:
        msg = "Expected R, T to have the same batch dimension; got %r, %r"
        raise ValueError(msg % (R.shape[0], T.shape[0]))
    if T.dim() != 2 or T.shape[1:] != (3,):
        msg = "Expected T to have shape (N, 3); got %r"
        raise ValueError(msg % repr(T.shape))
    if R.dim() != 3 or R.shape[1:] != (3, 3):
        msg = "Expected R to have shape (N, 3, 3); got %r"
        raise ValueError(msg % repr(R.shape))

    # Create a Transform3d object
    T = Translate(T, device=T.device)
    R = Rotate(R, device=R.device)
    return R.compose(T)
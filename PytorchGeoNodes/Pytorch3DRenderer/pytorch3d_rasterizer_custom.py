#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from pytorch3d.renderer.cameras import get_world_to_view_transform
from pytorch3d.renderer.points.rasterize_points import rasterize_points
from pytorch3d.renderer.points.rasterizer import PointFragments, PointsRasterizationSettings
from pytorch3d.renderer.mesh.rasterizer import Fragments, RasterizationSettings
from pytorch3d.renderer.mesh.rasterize_meshes import rasterize_meshes
from pytorch3d.renderer.mesh.utils import _clip_barycentric_coordinates, _interpolate_zbuf


from .pytorch3d_camera_custom import transform_scannet_points

class PointsRasterizerScanNet(nn.Module):
    """
    This class implements methods for rasterizing a batch of pointclouds.
    """

    def __init__(self, cameras, raster_settings=None):
        """
        cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-screen
                transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = PointsRasterizationSettings()

        self.cameras = cameras
        self.raster_settings = raster_settings

    def custom_offset(self, pointclouds, offsets_packed):
        """
        Translate the point clouds by an offset. In place operation.

        Args:
            offsets_packed: A Tensor of the same shape as self.points_packed
                giving offsets to be added to all points.
        Returns:
            self.
        """
        points_packed = pointclouds.points_packed()
        points_packed = transform_scannet_points(points_packed)
        if offsets_packed.shape != points_packed.shape:
            raise ValueError("Offsets must have dimension (all_p, 3).")
        pointclouds._points_packed = points_packed + offsets_packed
        new_points_list = list(
            pointclouds._points_packed.split(pointclouds.num_points_per_cloud().tolist(), 0)
        )
        # Note that since _compute_packed() has been executed, points_list
        # cannot be None even if not provided during construction.
        pointclouds._points_list = new_points_list
        if pointclouds._points_padded is not None:
            for i, points in enumerate(new_points_list):
                if len(points) > 0:
                    pointclouds._points_padded[i, : points.shape[0], :] = points
        return pointclouds

    def offset(self, pointclouds, offsets_packed):
        """
        Out of place offset.

        Args:
            offsets_packed: A Tensor of the same shape as self.points_packed
                giving offsets to be added to all points.
        Returns:
            new Pointclouds object.
        """
        new_clouds = pointclouds.clone()
        return self.custom_offset(new_clouds, offsets_packed)

    def transform(self, point_clouds, **kwargs) -> torch.Tensor:
        """
        Args:
            point_clouds: a set of point clouds

        Returns:
            points_screen: the points with the vertex positions in screen
            space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """
        cameras = kwargs.get("cameras", self.cameras)

        pts_world = point_clouds.points_padded()
        pts_world_packed = point_clouds.points_packed()
        pts_world_packed = transform_scannet_points(pts_world_packed)

        pts_screen = cameras.transform_points(pts_world, **kwargs)
        pts_screen = transform_scannet_points(pts_screen)

        # NOTE: Retaining view space z coordinate for now.
        # TODO: Remove this line when the convention for the z coordinate in
        # the rasterizer is decided. i.e. retain z in view space or transform
        # to a different range.
        view_transform = get_world_to_view_transform(R=cameras.R, T=cameras.T)
        verts_view = view_transform.transform_points(pts_world)
        verts_view = transform_scannet_points(verts_view)
        pts_screen[..., 2] = verts_view[..., 2]

        # Offset points of input pointcloud to reuse cached padded/packed calculations.
        pad_to_packed_idx = point_clouds.padded_to_packed_idx()
        pts_screen_packed = pts_screen.view(-1, 3)[pad_to_packed_idx, :]
        pts_packed_offset = pts_screen_packed - pts_world_packed
        point_clouds = self.offset(point_clouds, pts_packed_offset)
        return point_clouds

    def forward(self, point_clouds, **kwargs) -> PointFragments:
        """
        Args:
            point_clouds: a set of point clouds with coordinates in world space.
        Returns:
            PointFragments: Rasterization outputs as a named tuple.
        """
        points_screen = self.transform(point_clouds, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        idx, zbuf, dists2 = rasterize_points(
            points_screen,
            image_size=raster_settings.image_size,
            radius=raster_settings.radius,
            points_per_pixel=raster_settings.points_per_pixel,
            bin_size=raster_settings.bin_size,
            max_points_per_bin=raster_settings.max_points_per_bin,
        )
        return PointFragments(idx=idx, zbuf=zbuf, dists=dists2)


class MeshRasterizerScanNet(nn.Module):
    """
    This class implements methods for rasterizing a batch of heterogenous
    Meshes.
    """

    def __init__(self, cameras, raster_settings=None):
        """
        Args:
            cameras: A cameras object which has a  `transform_points` method
                which returns the transformed points after applying the
                world-to-view and view-to-screen
                transformations.
            raster_settings: the parameters for rasterization. This should be a
                named tuple.

        All these initial settings can be overridden by passing keyword
        arguments to the forward function.
        """
        super().__init__()
        if raster_settings is None:
            raster_settings = RasterizationSettings()

        self.cameras = cameras
        self.raster_settings = raster_settings

    def transform(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                vertex coordinates in world space.

        Returns:
            meshes_screen: a Meshes object with the vertex positions in screen
            space

        NOTE: keeping this as a separate function for readability but it could
        be moved into forward.
        """

        import time
        cameras = kwargs.get("cameras", self.cameras)

        verts_world = meshes_world.verts_padded()
        verts_world_packed = meshes_world.verts_packed()
        start_time = time.time()
        verts_screen = cameras.transform_points(verts_world, **kwargs)
        print("transform points 1: ", time.time() - start_time)

        start_time = time.time()
        # NOTE: Retaining view space z coordinate for now.
        # TODO: Revisit whether or not to transform z coordinate to [-1, 1] or
        # [0, 1] range.
        view_transform = get_world_to_view_transform(R=cameras.R, T=cameras.T)
        verts_view = view_transform.transform_points(verts_world)
        verts_screen[..., 2] = verts_view[..., 2]
        print("transform points 2: ", time.time() - start_time)


        start_time = time.time()
        # Offset verts of input mesh to reuse cached padded/packed calculations.
        pad_to_packed_idx = meshes_world.verts_padded_to_packed_idx()
        verts_screen_packed = verts_screen.view(-1, 3)[pad_to_packed_idx, :]
        verts_packed_offset = verts_screen_packed - verts_world_packed
        print("offset 1: ", time.time() - start_time)

        start_time = time.time()
        meshes_view = meshes_world.offset_verts(verts_packed_offset)
        print("offset 2: ", time.time() - start_time)

        return meshes_view

    def forward(self, meshes_world, **kwargs) -> Fragments:
        """
        Args:
            meshes_world: a Meshes object representing a batch of meshes with
                          coordinates in world space.
        Returns:
            Fragments: Rasterization outputs as a named tuple.
        """
        meshes_screen = self.transform(meshes_world, **kwargs)
        raster_settings = kwargs.get("raster_settings", self.raster_settings)
        # TODO(jcjohns): Should we try to set perspective_correct automatically
        # based on the type of the camera?
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
            cull_backfaces=raster_settings.cull_backfaces,
        )
        return Fragments(
            pix_to_face=pix_to_face, zbuf=zbuf, bary_coords=bary_coords, dists=dists
        )


class MeshRendererScannet(nn.Module):
    """
    A class for rendering a batch of heterogeneous meshes. The class should
    be initialized with a rasterizer and shader class which each have a forward
    function.
    """

    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        """
        Render a batch of images from a batch of meshes by rasterizing and then
        shading.

        NOTE: If the blur radius for rasterization is > 0.0, some pixels can
        have one or more barycentric coordinates lying outside the range [0, 1].
        For a pixel with out of bounds barycentric coordinates with respect to a
        face f, clipping is required before interpolating the texture uv
        coordinates and z buffer so that the colors and depths are limited to
        the range for the corresponding face.
        """

        import time

        start_time = time.time()
        fragments = self.rasterizer(meshes_world, **kwargs)
        clipped_zbuf = fragments.zbuf
        images = self.shader(fragments, meshes_world, **kwargs)

        return images, clipped_zbuf
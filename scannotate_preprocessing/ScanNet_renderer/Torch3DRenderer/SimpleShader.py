import torch
import torch.nn as nn
from pytorch3d.renderer.mesh.shader import hard_rgb_blend
from pytorch3d.renderer.cameras import OpenGLPerspectiveCameras
from typing import Sequence
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes

# from pytorch3d.renderer.mesh.textures import interpolate_texture_map, interpolate_vertex_colors,interpolate_vertex_uvs,interpolate_vertex_normals
#from texturing import interpolate_texture_map, interpolate_vertex_colors,interpolate_vertex_uvs,interpolate_vertex_normals
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)

from pytorch3d.renderer.mesh.shader import ShaderBase, BlendParams


# class SimpleShader(nn.Module):
#     def __init__(self, device="cpu", cameras=None):
#         super().__init__()
#
#         self.cameras = (
#             cameras if cameras is not None else OpenGLPerspectiveCameras(device=device)
#         )
#     def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
#         cameras = kwargs.get("cameras", self.cameras)
#
#         pixel_colors = assign_vertex_colors(fragments, meshes)
#
#         images = hard_rgb_blend(pixel_colors, fragments)
#
#         return images # (N, H, W, 3) RGBA image
#
# def assign_vertex_colors(fragments, meshes) -> torch.Tensor:
#     """
#     Detemine the color for each rasterized face. Interpolate the colors for
#     vertices which form the face using the barycentric coordinates.
#     Args:
#         meshes: A Meshes class representing a batch of meshes.
#         fragments:
#             The outputs of rasterization. From this we use
#
#             - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
#               of the faces (in the packed representation) which
#               overlap each pixel in the image.
#             - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
#               the barycentric coordianates of each pixel
#               relative to the faces (in the packed
#               representation) which overlap the pixel.
#
#     Returns:
#         texels: An texture per pixel of shape (N, H, W, K, C).
#         There will be one C dimensional value for each element in
#         fragments.pix_to_face.
#     """
#     vertex_textures = meshes.textures.verts_features_packed().reshape(-1, 3)  # (V, C)
#     vertex_textures = vertex_textures[meshes.verts_padded_to_packed_idx(), :]
#     faces_packed = meshes.faces_packed()
#     faces_textures = vertex_textures[faces_packed]  # (F, 3, C)
#
#     texels = get_face_attributes(
#         fragments.pix_to_face, faces_textures
#     )
#
#     return texels
#
# def get_face_attributes(
#     pix_to_face: torch.Tensor,
#     face_attributes: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     Interpolate arbitrary face attributes using the barycentric coordinates
#     for each pixel in the rasterized output.
#
#     Args:
#         pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
#             of the faces (in the packed representation) which
#             overlap each pixel in the image.
#         face_attributes: packed attributes of shape (total_faces, 3, D),
#             specifying the value of the attribute for each
#             vertex in the face.
#
#     Returns:
#         pixel_vals: tensor of shape (N, H, W, K, D) giving the interpolated
#         value of the face attribute for each pixel.
#     """
#     F, FV, D = face_attributes.shape
#     if FV != 3:
#         raise ValueError("Faces can only have three vertices; got %r" % FV)
#     N, H, W, K = pix_to_face.shape
#     if pix_to_face.shape != (N, H, W, K):
#         msg = "pix_to_face must have shape (batch_size, H, W, K); got %r"
#         raise ValueError(msg % (pix_to_face.shape,))
#
#     empty_face = torch.cuda.FloatTensor(1, 3, 3).fill_(0)
#     face_attributes = torch.cat([face_attributes, empty_face], dim=0)
#     pixel_vals = face_attributes[pix_to_face[:,:,:,0],:,0]
#
#     pixel_vals = pixel_vals.view(N, H, W, 1, 3)
#     return pixel_vals

class VertexColorShader(ShaderBase):
    def __init__(self, blend_soft=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.blend_soft = blend_soft

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        if self.blend_soft:
            return softmax_rgb_blend(texels, fragments, blend_params)
        else:
            return hard_rgb_blend(texels, fragments, blend_params)

class SimpleShader(nn.Module):
    def __init__(self, device="cpu", cameras=None):
        super().__init__()

        self.cameras = (
            cameras if cameras is not None else OpenGLPerspectiveCameras(device=device)

        )
        self.blend_params = BlendParams(background_color= [0, 0, 0])

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)

        texels = meshes.sample_textures(fragments)

        images = hard_rgb_blend(texels, fragments, self.blend_params)

        return images # (N, H, W, 3) RGBA image

class UVsCorrespondenceShader(nn.Module):
    """
    UV correspondence shader will render the model with a custom texture map as it's input.
    No lightning or blending will be applied
    To use the default values, simply initialize the shader with the desired
    device e.g.
    .. code-block::
        shader = UVsCorrespondenceShader(
                blend_params=bp,
                device=device,
                cameras=cameras,
                colormap_padded=colormap_padded
    """

    def __init__(
            self, device="cpu", cameras=None, blend_params=None, colormap=None
    ):
        super().__init__()

        self.cameras = cameras
        self.colormap = colormap
        #self.blend_params = blend_params if blend_params is not None else BlendParams(sigma=0., gamma=0., background_color= [0, 0, 0])
        self.blend_params = blend_params if blend_params is not None else BlendParams(background_color= [0, 0, 0])

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        colormap = kwargs.get("colormap", self.colormap)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)
        # texels = interpolate_texture_map(fragments, meshes,colormap)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, self.blend_params)
        # images = softmax_rgb_blend(texels, fragments, self.blend_params)
        return images
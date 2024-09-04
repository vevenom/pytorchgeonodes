import torch
import torch.nn as nn
from pytorch3d.renderer.mesh.shader import hard_rgb_blend
from typing import Sequence

from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from typing import NamedTuple, Sequence, Union

def softmax_textureless_rgb_blend(
    fragments,
    blend_params: BlendParams,
    znear: Union[float, torch.Tensor] = 1.0,
    zfar: Union[float, torch.Tensor] = 100,
) -> torch.Tensor:
    """
    Adapted from softmax_rgb_blend from Pytorch3D.

    Textureless blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
        relative z distances.
      - **A** - blend based on the 2D distance based probability map.

    Args:
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.
            - zbuf: FloatTensor of shape (N, H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping faces.
        blend_params: instance of BlendParams dataclass containing properties
            - sigma: float, parameter which controls the width of the sigmoid
              function used to calculate the 2D distance based probability.
              Sigma controls the sharpness of the edges of the shape.
            - gamma: float, parameter which controls the scaling of the
              exponential function used to control the opacity of the color.
            - background_color: (3) element list/tuple/torch.Tensor specifying
              the RGB values for the background color.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction

    Returns:
        RGBA pixel_colors: (N, H, W, 4)

    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """

    N, H, W, K = fragments.pix_to_face.shape
    pixel_colors = torch.ones((N, H, W, 4), dtype=torch.float32, device=fragments.pix_to_face.device)
    background_color_ = blend_params.background_color
    if isinstance(background_color_, torch.Tensor):
        background_color = background_color_.to(fragments.pix_to_face.device)
    else:
        background_color = torch.tensor(background_color_,
                                        dtype=torch.float32,
                                        device=fragments.pix_to_face.device)

    # Weight for background color
    eps = 1e-10

    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    alpha = torch.prod((1.0 - prob_map), dim=-1)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    # TODO: there may still be some instability in the exponent calculation.

    # Reshape to be compatible with (N, H, W, K) values in fragments
    if torch.is_tensor(zfar):
        # pyre-fixme[16]
        zfar = zfar[:, None, None, None]
    if torch.is_tensor(znear):
        # pyre-fixme[16]: Item `float` of `Union[float, Tensor]` has no attribute
        #  `__getitem__`.
        znear = znear[:, None, None, None]

    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

    # Also apply exp normalize trick for the background color weight.
    # Clamp to ensure delta is never 0.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta

    # Sum: weights * textures + background color
    weighted_colors = (weights_num[..., None] * torch.ones_like(background_color)).sum(dim=-2)
    weighted_background = delta * background_color
    pixel_colors[..., :3] = (weighted_colors + weighted_background) / denom
    pixel_colors[..., 3] = 1.0 - alpha

    return pixel_colors

class SimpleTexturelessShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:

        images = softmax_textureless_rgb_blend(fragments, blend_params=BlendParams(background_color= [0, 0, 0]))

        return images # (N, H, W, 3) RGBA image

class SimpleShader(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:

        texels = meshes.sample_textures(fragments)

        images = hard_rgb_blend(texels, fragments, blend_params=BlendParams(background_color= [0, 0, 0]))

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
        # self.blend_params = blend_params if blend_params is not None else BlendParams(sigma=0., gamma=0., background_color= [0, 0, 0])
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
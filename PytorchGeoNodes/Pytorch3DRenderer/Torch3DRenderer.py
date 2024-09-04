import torch
from pytorch3d.renderer.cameras import PerspectiveCameras

from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
)

from PytorchGeoNodes.Pytorch3DRenderer.pytorch3d_rasterizer_custom import MeshRendererScannet
from PytorchGeoNodes.Pytorch3DRenderer.SimpleShader import SimpleTexturelessShader, SimpleShader

class Torch3DRenderer:
    def __init__(self):

        self.device = device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    def create_pytorch3d_rasterization_settings(self, image_size, z_clip_value):
        return RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
            perspective_correct=True,
            clip_barycentric_coords=False,
            cull_backfaces=False,
            z_clip_value=z_clip_value
        )

    def create_renderer(self, pose, intrinsics, image_size, z_clip_value, texturless, device):
        raster_settings = self.create_pytorch3d_rasterization_settings(image_size, z_clip_value)

        fx = intrinsics[0, 0, 0]
        fy = intrinsics[0, 1, 1]
        px = intrinsics[0, 0, 2]
        py = intrinsics[0, 1, 2]
        focal_length = (
            torch.tensor([fx, fy])[None].type(torch.FloatTensor).to(device))
        principal_point = torch.tensor([px, py])[None].type(torch.FloatTensor).to(device)

        cameras = PerspectiveCameras(
            focal_length=focal_length,
            # focal_length=4,
            principal_point=principal_point,
            in_ndc=False,
            device=device,
            T=pose[:, :3, 3],
            R=pose[:, :3, :3],
            image_size=(image_size,))

        if texturless:
            shader = SimpleTexturelessShader()

            renderer = MeshRendererScannet(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
                ),
                shader=shader
            )
        else:
            shader = SimpleShader()
            lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
            blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings),
                shader=SoftPhongShader(
                    device=device, cameras=cameras, lights=lights, blend_params=blend_params
                ),
            )

        return renderer






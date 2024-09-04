import numpy as np
import torch
import torch.nn as nn
import os
import shutil
import matplotlib.pyplot as plt
import argparse

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)

from PytorchGeoNodes.GeometryNodes import GeometryNodes
from PytorchGeoNodes.BlenderShapeProgram import BlenderShapeProgram

shape_program = BlenderShapeProgram(config_path='configs_shape_programs/sp_cabinet.json')
params_tree = shape_program.parse_params_tree_()
params_tree.set_values_from_dict({
    'Width': 1.050,
    'Board Thickness': 0.04,
    'Depth': 0.6,
    'Height': 0.98,
    'Number of Dividing Boards': 3,
    'Dividing Board Thickness': 0.04,
    'Has Drawers': False,
    'Has Back': False,
    'Has Legs': True,
    'Leg Width': 0.04,
    'Leg Height': 0.04,
    'Leg Depth': 0.04
}, is_normalized=False)
shape_program.set_params_from_tree(params_tree)

device = torch.device("cuda:0")

# input_params_dict = {}
input_params_dict = {
    'Depth': nn.Parameter(torch.tensor([[1.0]], device=device)),
    'Height': nn.Parameter(torch.tensor([[1.6]], device=device))}

geometry_nodes = GeometryNodes(shape_program)
geometry_nodes.to(device)

# render non-diff using pytorch3d
non_diff_py3d_mesh = shape_program.py3d_get_mesh().to(device)

R, T = look_at_view_transform(2.7, 0, 0, device=device)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
    faces_per_pixel=100,
)
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params)
)
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1,
)
# We can add a point light in front of the object.
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
)
optim = torch.optim.Adam(list(input_params_dict.values()))

def loss_fn(pred_mesh, non_diff_py3d_mesh, R, T, get_image=False):
    pred_image = silhouette_renderer(meshes_world=pred_mesh, R=R, T=T)

    gt_image = silhouette_renderer(meshes_world=non_diff_py3d_mesh, R=R, T=T)

    loss = torch.sum((pred_image - gt_image) ** 2)

    if get_image:
        return loss, pred_image, gt_image
    return loss

@torch.no_grad()
def log(pred_mesh, non_diff_py3d_mesh, R, T):
    loss, pred_image, gt_image = loss_fn(pred_mesh, non_diff_py3d_mesh, R, T, get_image=True)

    # convert to numpy
    pred_image_np = pred_image.detach().cpu().numpy()[0][..., 3]
    gt_image_np = gt_image.detach().cpu().numpy()[0][..., 3]
    abs_diff = np.abs(pred_image_np - gt_image_np)

    print('iter', iter, 'loss', loss.item())

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(pred_image_np)
    plt.subplot(1, 3, 2)
    plt.imshow(gt_image_np)
    plt.subplot(1, 3, 3)
    plt.imshow(abs_diff)
    # plt.show()
    plt.savefig(os.path.join(demo_dir, '%04d.png' % iter))
    plt.close()

parser = argparse.ArgumentParser(description='Reconstruct objects')
parser.add_argument('--experiment_path', type=str, help='Dataset path')
args = parser.parse_args()

demo_dir = args.experiment_path
if os.path.exists(demo_dir):
    shutil.rmtree(demo_dir)
os.makedirs(demo_dir)

for iter in range(1000):
    # randomize dist, elev, azim
    dist = np.random.uniform(2.5, 3.5)
    elev = np.random.uniform(-np.pi / 8, np.pi / 8)
    azim = np.random.uniform(-np.pi / 8, np.pi / 8)

    R, T = look_at_view_transform(dist, elev, azim, degrees=False, device=device)

    _, outputs = geometry_nodes.forward(input_params_dict, transform2blender_coords=True)
    pred_mesh = outputs[0][0][0]

    loss = loss_fn(pred_mesh, non_diff_py3d_mesh, R, T)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if iter % 100 == 0:
        log(pred_mesh, non_diff_py3d_mesh, R, T)













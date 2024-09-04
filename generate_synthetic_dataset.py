import numpy as np
import torch
import os
import pickle
import argparse

from pytorch3d.renderer import (
    look_at_view_transform, )
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.transforms import axis_angle_to_matrix, Rotate

from PytorchGeoNodes.Pytorch3DRenderer.Torch3DRenderer import Torch3DRenderer

from PytorchGeoNodes.GeometryNodes import GeometryNodes
from PytorchGeoNodes.BlenderShapeProgram import BlenderShapeProgram

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_renderer():
    num_views = 10
    azi = np.linspace(0, 2 * np.pi, num_views)
    azi = torch.tensor(azi, dtype=torch.float32, device=device).view(-1, 1)

    dists = torch.tensor([3.0], dtype=torch.float32, device=device)
    dists = dists.repeat(num_views, 1)
    elev = torch.tensor([0.3], dtype=torch.float32, device=device)
    elev = elev.repeat(num_views, 1)

    R, t = look_at_view_transform(dists, elev, azi, degrees=False, device=device)
    pose = torch.zeros((num_views, 4, 4), dtype=torch.float32, device=device)
    pose[:, :3, :3] = R
    pose[:, :3, 3] = t

    intrinsics = np.array([[577.590698, 0.000000, 318.905426, 0.000000],
                           [0.000000, 578.729797, 242.683609, 0.000000],
                           [0.000000, 0.000000, 1.000000, 0.000000],
                           [0.000000, 0.000000, 0.000000, 1.000000]])
    intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=device)
    intrinsics = intrinsics.view(1, 4, 4)
    image_size = (640, 480)
    scale = 2.0

    intrinsics[:, 0, 0] /= scale
    intrinsics[:, 1, 1] /= scale
    intrinsics[:, 0, 2] /= scale
    intrinsics[:, 1, 2] /= scale

    image_size = (int(image_size[0] / scale), int(image_size[1] / scale))

    renderer_textureless = Torch3DRenderer().create_renderer(pose, intrinsics,
                                                             image_size,
                                                             z_clip_value=0.1,
                                                             texturless=True,
                                                             device=device)

    return renderer_textureless

renderer = create_renderer()

skip_existing_reconstructions = True

if __name__ == '__main__':

    # parse input args
    # -- category
    # -- server / local

    parser = argparse.ArgumentParser(description='Generate synthetic dataset')
    parser.add_argument('--category', type=str, default='cabinet', help='Object category')
    parser.add_argument('--num_scenes', type=int, default=100, help='Number of scenes to generate')
    parser.add_argument('--dataset_path', type=str, help='Dataset path')
    args = parser.parse_args()

    synthetic_dataset_path = args.dataset_path

    object_category = args.category
    num_scenes = args.num_scenes

    if object_category == 'cabinet':
        shape_program = BlenderShapeProgram(config_path='configs_shape_programs/sp_synth_cabinet.json')
    else:
        shape_program = BlenderShapeProgram(config_path='configs_shape_programs/sp_' + object_category + '.json')
    params_tree = shape_program.parse_params_tree_()

    geometry_nodes = GeometryNodes(shape_program)
    geometry_nodes.to(device)

    for scene_ind in range(num_scenes):
        print('Generating scene', scene_ind)

        # create a new scene with scene_ind formatted as 0000, 0001, 0002, ...
        scene_name = str(scene_ind).zfill(4)
        scene_path = os.path.join(synthetic_dataset_path, object_category, scene_name)
        if os.path.exists(scene_path):
            continue
        os.makedirs(scene_path)

        params_tree.randomize_tree_values()

        shape_program.set_params_from_tree(params_tree)

        input_params_dict = {}
        sp_params = params_tree.to_params_dict(get_normalized_values=False)
        for param_name, value in sp_params.items():
            input_params_dict[param_name] = torch.tensor([[value]], device=device)

        _, outputs = geometry_nodes.forward(input_params_dict, transform2blender_coords=True)
        mesh = outputs[0][0][0] #

        random_rotation_y = np.random.uniform(0, 2 * np.pi)
        random_rotation_y = torch.tensor(random_rotation_y, dtype=torch.float32, device=device)

        rotation_matrix = axis_angle_to_matrix(torch.tensor([0, random_rotation_y, 0],
                                                            dtype=torch.float32, device=device))
        rotation_matrix = rotation_matrix[None]
        transf = Rotate(R=rotation_matrix)

        verts = mesh.verts_packed()

        verts = transf.transform_points(verts)
        mesh = Meshes(verts=[verts], faces=[mesh.faces_packed()], textures=mesh.textures)


        n_views = renderer.rasterizer.cameras.T.shape[0]

        # extend the mesh to have a batch dimension
        mesh = mesh.extend(n_views)  # type: Meshes

        # 2. render the scene
        # render the scene
        rendered_mesh, zbuf = renderer(mesh)
        rendered_mesh = rendered_mesh[..., 1]

        mesh_mask = rendered_mesh != 0

        depth_pred = zbuf[..., 0]
        depth_pred[depth_pred < 0] = 0

        # 3. sample pcd
        # sample pcd
        pcd = sample_points_from_meshes(mesh[0], num_samples=10000)

        # create final dict
        scene_dict = {
            'scene_name': scene_name,
            'scene_path': scene_path,
            'scene_pcd': pcd,
            'scene_mask': mesh_mask,
            'scene_depth': depth_pred,
            'sp_params': sp_params,
            'rotation_y': random_rotation_y,
            'renderer': renderer
        }

        # Export pickle
        with open(os.path.join(scene_path, 'scene_dict.pkl'), 'wb') as f:
            pickle.dump(scene_dict, f)



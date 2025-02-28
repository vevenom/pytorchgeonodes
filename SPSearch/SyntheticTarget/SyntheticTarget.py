import torch
import numpy as np
import os
import copy

from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import Translate
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss.chamfer import chamfer_distance

from SPSearch.Target import Target

class SyntheticTarget(Target):
    def __init__(self, scene_dict,
                 geometry_nodes,
                 log_path=None,
                 settings=None):
        super().__init__(log_path)

        assert log_path is not None, "log_path must be provided"
        self.log_path = log_path

        self.geometry_nodes = geometry_nodes

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.optimize_translation = False

        self.scene_name = scene_dict['scene_name']
        self.scene_mask = scene_dict['scene_mask']
        self.scene_depth = scene_dict['scene_depth']
        self.renderer = scene_dict['renderer']
        self.scene_pcd = scene_dict['scene_pcd']

        print('SytheticTarget initialized with scene: {}'.format(scene_dict['scene_name']))

    def get_input_attributes(self):
        scene_dict = {
            'scene_name': self.scene_name,
            'scene_pcd': self.scene_pcd,
            'scene_mask': self.scene_mask,
            'scene_depth': self.scene_depth,
            'renderer': self.renderer
        }
        return scene_dict, self.geometry_nodes, self.log_path

    def loss(self, obj_mesh, scene_dict):

        mesh_pcd = (
            sample_points_from_meshes(obj_mesh, num_samples=10000))
        mesh_pcd = torch.cat((mesh_pcd, obj_mesh.verts_packed()[None]), dim=1)

        (cd_loss_x, cd_loss_y) = chamfer_distance(scene_dict['scene_pcd'], mesh_pcd,
                                                  batch_reduction=None,
                                                  point_reduction=None,
                                                  norm=1,
                                                  single_directional=False)[0]  # / self.cd_clamp

        loss = cd_loss_x.mean() + cd_loss_y.mean()

        return loss


    def calculate_mesh_from_input_dict(self, input_params_dict, rotation_matrix, translation_offset=None):
        device = self.device

        _, outputs = self.geometry_nodes.forward(input_params_dict, transform2blender_coords=True)
        obj_mesh = outputs[0][0][0]
        obj_mesh = Meshes(verts=obj_mesh.verts, faces=obj_mesh.faces)
        verts = obj_mesh.verts_packed()
        faces = obj_mesh.faces_packed()

        bb = obj_mesh.get_bounding_boxes()  # (N, 3, 2)
        bb_center = (bb[:, :, 1] - bb[:, :, 0]) / 2 + bb[:, :, 0]

        verts = verts - bb_center

        transform = Transform3d(device=device)

        if rotation_matrix is not None:
            transform_rot = Transform3d(matrix=rotation_matrix, device=device)
            assert rotation_matrix.shape[0] == 1
            transform = transform.compose(transform_rot)
        else:
            assert False, "We are doing experiments with rotations"

        verts = transform.transform_points(verts)

        obj_mesh = Meshes(verts=[verts], faces=[faces], textures=obj_mesh.textures)

        if translation_offset is not None:
            translate = Translate(translation_offset)
            verts = translate.transform_points(verts)

        obj_mesh = Meshes(verts=[verts], faces=[faces], textures=obj_mesh.textures)

        return obj_mesh

    def calculate_cost_from_input_dict(self, input_params_dict, rotation_matrix, translation_offset=None):

        obj_mesh = self.calculate_mesh_from_input_dict(input_params_dict, rotation_matrix, translation_offset)

        scene_dict = {
            'scene_pcd': self.scene_pcd,
            'scene_mask': self.scene_mask,
            'scene_depth': self.scene_depth,
            'renderer': self.renderer
        }

        loss = self.loss(obj_mesh, scene_dict)

        return loss

    @torch.no_grad()
    def log_iter_from_input_dict(self, input_params_dict, rotation_matrix,
                                 translation_offset=None, iter_num=0, file_prefix=''):
        device = self.device

        obj_mesh = self.calculate_mesh_from_input_dict(input_params_dict, rotation_matrix, translation_offset)

        num_views = self.renderer.rasterizer.cameras.T.shape[0]
        obj_mesh = obj_mesh.extend(num_views)

        # self.scene_dict contains:
        # {
        #     'scene_pcd',
        #     'scene_mask',
        #     'scene_depth',
        #     'renderer'
        # }

        # render the scene
        rendered_mesh, zbuf = self.renderer(obj_mesh)

        rendered_mesh = rendered_mesh[..., 1]

        mesh_mask = rendered_mesh != 0

        depth_pred = zbuf[..., 0]

        for i in range(num_views):
            depth_pred_i = depth_pred[i].detach().cpu().numpy()
            depth_pred_i[depth_pred_i < 0] = 0

            depth_gt = self.scene_depth[i].detach().cpu().numpy()
            depth_gt[depth_gt < 0] = 0

            abs_diff = np.abs(depth_gt - depth_pred_i)

            vis_img = np.concatenate((depth_gt, depth_pred_i, abs_diff), axis=1)


            out_path = os.path.join(self.log_path, file_prefix + 'vis_{:05d}_'.format(iter_num) +
                                     str(i) + '_' + '.jpg')
            # cv2.imwrite(out_path, vis_img)

            # save vis_img using matplotlib
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(vis_img)
            ax.axis('off')
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

    def render_image(self, input_params_dict, rotation_matrix,
                                 translation_offset=None, image_ind=0):
        obj_mesh = self.calculate_mesh_from_input_dict(input_params_dict, rotation_matrix, translation_offset)

        renderer = copy.deepcopy(self.renderer)
        renderer.rasterizer.cameras = renderer.rasterizer.cameras[image_ind]
        num_views = 1
        obj_mesh = obj_mesh.extend(num_views)

        # render the scene
        rendered_mesh, zbuf = renderer(obj_mesh)

        rendered_mesh = rendered_mesh[..., 1]

        mesh_mask = rendered_mesh != 0

        depth_pred = zbuf[..., 0]

        depth_pred_i = depth_pred[0].detach().cpu().numpy()

        return depth_pred_i

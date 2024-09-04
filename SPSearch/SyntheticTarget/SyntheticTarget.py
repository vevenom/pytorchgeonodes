import torch
import numpy as np
import os

from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import Translate
from pytorch3d.structures import Meshes

from SPSearch.Target import Target
from SPSearch.SyntheticTarget.LossBatched import LossBatched


class SyntheticTarget(Target):
    def __init__(self, scene_dict,
                 geometry_nodes,
                 log_path=None):
        super().__init__(log_path)

        assert log_path is not None, "log_path must be provided"
        self.log_path = log_path

        self.geometry_nodes = geometry_nodes

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.loss = LossBatched()

        self.scene_dict = scene_dict

        print('SytheticTarget initialized with scene: {}'.format(scene_dict['scene_name']))

    def calculate_mesh_from_input_dict(self, input_params_dict, rotation_matrix, translation_offset=None):
        device = self.device

        #         start.record()
        _, outputs = self.geometry_nodes.forward(input_params_dict, transform2blender_coords=True)
        obj_mesh = outputs[0][0][0]
        #         end.record()

        #         torch.cuda.synchronize()

        # print('Time needed for geometry nodes: {}'.format(start.elapsed_time(end)))

        verts = obj_mesh.verts_packed()
        faces = obj_mesh.faces_packed()

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
        device = self.device

        # calculate time needed for loss function
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        obj_mesh = self.calculate_mesh_from_input_dict(input_params_dict, rotation_matrix, translation_offset)

#         start.record()
#         loss = self.scannotate_loss(obj_mesh, scene_frame_iterator,
#                                     self.scan2cad_instance.scannet_instance.scannet_pose_to_py3d)

        loss = self.loss(obj_mesh, self.scene_dict)

#         end.record()

#         torch.cuda.synchronize()
        # print('Time needed for loss function: {}'.format(start.elapsed_time(end)))

        return loss

    @torch.no_grad()
    def log_iter_from_input_dict(self, input_params_dict, rotation_matrix,
                                 translation_offset=None, iter_num=0, file_prefix=''):
        device = self.device

        obj_mesh = self.calculate_mesh_from_input_dict(input_params_dict, rotation_matrix, translation_offset)

        num_views = self.scene_dict['renderer'].rasterizer.cameras.T.shape[0]
        obj_mesh = obj_mesh.extend(num_views)

        # self.scene_dict contains:
        # {
        #     'scene_pcd',
        #     'scene_mask',
        #     'scene_depth',
        #     'renderer'
        # }

        # render the scene
        rendered_mesh, zbuf = self.scene_dict['renderer'](obj_mesh)

        rendered_mesh = rendered_mesh[..., 1]

        mesh_mask = rendered_mesh != 0

        depth_pred = zbuf[..., 0]

        for i in range(num_views):
            depth_pred_i = depth_pred[i].detach().cpu().numpy()
            depth_pred_i[depth_pred_i < 0] = 0

            depth_gt = self.scene_dict['scene_depth'][i].detach().cpu().numpy()
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
            plt.savefig(out_path)
            plt.close(fig)

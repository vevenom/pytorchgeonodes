import numpy as np
import cv2
import os
import open3d as o3d
import copy
import torch
from utils import SEMANTIC_IDX2NAME, COLOR_DETECTRON2


def load_depth_img(path):
    depth_image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 1000.
    return depth_image

def load_rgb_img(path):
    rgb_img = cv2.imread(path)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    rgb_img = rgb_img / 255.
    return rgb_img

def save_depth_img(depth,depth_out_path,filename=None):
    if depth.ndim == 4:
        depth = depth[:, :, :, 0]

    depth_im_out = None

    for cnt, depth_im in enumerate(depth):
        #max_depth = 4.818183
        #depth_norm = ((depth_im /max_depth) * 255).astype('uint8')
        depth_norm = ((depth_im / np.max(depth_im)) * 255).astype('uint8')
        #cv2.imwrite(os.path.join(depth_out_path,  filename + '_' + str(cnt) + '_.png'), depth_norm)

        if depth_im_out is None:
            depth_im_out = depth_norm
        else:
            depth_im_out = np.vstack((depth_im_out, depth_norm))

    if filename is None:
        cv2.imwrite(os.path.join(depth_out_path, 'depth_rendered.png'), depth_im_out)
    else:
        cv2.imwrite(os.path.join(depth_out_path, filename + '.png'), depth_im_out)
    return

def save_normals_img(normals,depth_out_path):

    normals_im_out = None

    for cnt, normals_img in enumerate(normals):
        img = normals_img[:,:,0,:]

        img = (((img + 1. ) / 2.) * 255).astype('uint8')

        if normals_im_out is None:
            normals_im_out = img
        else:
            normals_im_out = np.vstack((normals_im_out, img))

    cv2.imwrite(os.path.join(depth_out_path, 'normals_rendered.png'), normals_im_out)

    return

def save_rgb_img(normals,depth_out_path,filename):

    normals_im_out = None

    for cnt, normals_img in enumerate(normals):
        img = normals_img

        rgb_img = (img * 255).astype('uint8')
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        if normals_im_out is None:
            normals_im_out = rgb_img
        else:
            normals_im_out = np.vstack((normals_im_out, rgb_img))

    if filename is None:
        cv2.imwrite(os.path.join(depth_out_path, 'depth_rendered.png'), normals_im_out)
    else:
        cv2.imwrite(os.path.join(depth_out_path, filename + '.png'), normals_im_out)
    return

def cut_meshes(mesh_o3d, indices_list,inst_label,scene_name):
    #mesh_points = tmesh.verts_packed()
    #mesh_faces = tmesh.faces_packed()

    #mesh_o3d = o3d.geometry.TriangleMesh()
    #points_o3d = mesh_points.detach().squeeze().cpu().numpy()
    #mesh_o3d.vertices = o3d.utility.Vector3dVector(points_o3d)
    #faces_o3d = mesh_faces.detach().squeeze().cpu().numpy()
    #mesh_o3d.triangles = o3d.utility.Vector3iVector(faces_o3d)
    mesh_o3d_obj = copy.deepcopy(mesh_o3d)
    mesh_o3d_obj = mesh_o3d_obj.select_by_index(indices_list)
    mesh_o3d.remove_vertices_by_index(indices_list)

    #path_tmp = os.path.join('/home/stefan/PycharmProjects/Scannet_Total3D/demo_output/cut_mesh/object.ply')
    #tmp = o3d.io.write_triangle_mesh(path_tmp, mesh_o3d_obj)

    #path_tmp = os.path.join('/home/stefan/PycharmProjects/Scannet_Total3D/demo_output/cut_mesh/background.ply')
    #tmp = o3d.io.write_triangle_mesh(path_tmp, mesh_o3d)

    #color_bg = np.ones_like(np.asarray(mesh_o3d.vertices)) * [-1., -1., -1.]
    #mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(color_bg)

    #color_obj = np.ones_like(np.asarray(mesh_o3d_obj.vertices)) * [inst_label/255., inst_label/255., inst_label/255.]
    #mesh_o3d_obj.vertex_colors = o3d.utility.Vector3dVector(color_obj)
    #mesh_o3d_obj.paint_uniform_color([inst_label/255., inst_label/255., inst_label/255.])


    face_list_bg = np.asarray(mesh_o3d.triangles)
    face_list_obj = np.asarray(mesh_o3d_obj.triangles)
    #tex_bg = torch.tensor(np.asarray(mesh_o3d.vertex_colors)).float()
    #tex_bg = tex_bg.unsqueeze(dim=0)
    #tex_bg = pytorch3d.renderer.mesh.textures.TexturesVertex(verts_features=tex_bg)



    mesh_bg = Meshes(
        verts=[torch.tensor(np.asarray(mesh_o3d.vertices)).float()],
        faces=[torch.tensor(np.asarray(face_list_bg))],
        #textures=tex_bg
    )
    #tex_obj = torch.tensor(np.asarray(mesh_o3d_obj.vertex_colors)).float()
    #tex_obj = tex_obj.unsqueeze(dim=0)
    #tex_obj = pytorch3d.renderer.mesh.textures.TexturesVertex(verts_features=tex_obj)

    mesh_obj = Meshes(
        verts=[torch.tensor(np.asarray(mesh_o3d_obj.vertices)).float()],
        faces=[torch.tensor(np.asarray(face_list_obj))],
        #textures=tex_obj
    )

    # path_tmp = os.path.join('/home/stefan/PycharmProjects/Scannet_Total3D/demo_output/render_test',
    #                  str(inst_label) + '_obj.ply')
    # tmp = o3d.io.write_triangle_mesh(path_tmp, mesh_o3d_obj)


    #mesh_obj = IO().load_mesh(path=path_tmp,include_textures=True)

    # IO().save_mesh(data=mesh_bg,
    #               path=os.path.join(
    #                   '/home/stefan/PycharmProjects/Scannet_Total3D/demo_output/cut_mesh','bg_.ply'),
    #                include_textures=False)
    #
    # IO().save_mesh(data=mesh_obj,
    #                path=os.path.join(
    #                    '/home/stefan/PycharmProjects/Scannet_Total3D/demo_output/cut_mesh',
    #                    'obj_.ply'),
    #                include_textures=False)
    return mesh_bg, mesh_obj


def drawOpen3dCylLines(bbListIn,col=None):
    # draw the BBs
    # lines = [[0, 1], [0, 2], [1, 3], [2, 3],
    #          [4, 5], [4, 6], [5, 7], [6, 7],
    #          [0, 4], [1, 5], [2, 6], [3, 7],
    #
    #          [1, 0], [2, 0], [3, 1], [3, 2],
    #          [5, 4], [6, 4], [7, 5], [7, 6],
    #          [4, 0], [5, 1], [6, 2], [7, 3]
    #          ]

    # for o3d
    #lines = [[0, 1],[0,2],[0,3],[3,5],[2,5],[4,5],[4,6],
    #         [3,6],[1,6],[2,7],[4,7],[1,7]]

    # for trimesh
    lines = [[0, 1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
             [0,4],[1,5],[2,6],[3,7]]

    line_sets = []

    for bb in bbListIn:
        points = bb
        if col is None:
            col = [0,0,1]
        colors = [col for i in range(len(lines))]

        line_mesh1 = LineMesh(points, lines, colors, radius=0.02)
        line_mesh1_geoms = line_mesh1.cylinder_segments
        line_sets = line_mesh1_geoms[0]
        for l in line_mesh1_geoms[1:]:
            line_sets = line_sets + l
            # line_sets.append(line_mesh1_geoms)

    return line_sets

def normalize_point(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def get_bdb_from_corners(corners,planexy=[3, 2, 6, 7]):
    """
    get coeffs, basis, centroid from corners
    :param corners: 8x3 numpy array
        corners of a 3D bounding box
    :return: bounding box parameters
    """
    up_max = np.max(corners[:, 1])
    up_min = np.min(corners[:, 1])

    points_2d = corners[planexy,:]
    points_2d = points_2d[np.argsort(points_2d[:, 0]), :]

    vector2 = np.array([points_2d[1, 0] - points_2d[0, 0], 0, points_2d[1, 2] - points_2d[0, 2]])
    vector1 = np.array([points_2d[2, 0] - points_2d[0, 0], 0, points_2d[2, 2] - points_2d[0, 2]])

    coeff1 = np.linalg.norm(vector1)
    coeff2 = np.linalg.norm(vector2)
    vector1 = normalize_point(vector1)
    vector2 = np.cross(vector1,[0,1,0])#normalize_point(vector2)
    centroid = np.array([points_2d[0, 0] + points_2d[3, 0], float(up_max) + float(up_min), points_2d[0, 2] + points_2d[3, 2]]) * 0.5

    basis = np.array([vector1,[0, 1, 0],vector2])
    coeffs = np.array([coeff1,up_max - up_min,coeff2]) * 0.5
    return centroid, basis.T, coeffs

def get_corners_of_bb3d_no_index(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    coeffs = np.abs(coeffs)
    corners[0, :] = + basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = - basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = - basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = + basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = + basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[5, :] = - basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[6, :] = - basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners[7, :] = + basis[0, :] * coeffs[0] - basis[1, :] * coeffs[1] - basis[2, :] * coeffs[2]
    corners = corners + np.tile(centroid, (8, 1))
    return corners

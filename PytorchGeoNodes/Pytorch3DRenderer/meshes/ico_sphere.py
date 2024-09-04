import torch
from pytorch3d.structures import Meshes

def create_icosphere(translation, scale, device):
    # https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/utils/ico_sphere.html
    ico_verts0 = [
        [-0.5257, 0.8507, 0.0000],
        [0.5257, 0.8507, 0.0000],
        [-0.5257, -0.8507, 0.0000],
        [0.5257, -0.8507, 0.0000],
        [0.0000, -0.5257, 0.8507],
        [0.0000, 0.5257, 0.8507],
        [0.0000, -0.5257, -0.8507],
        [0.0000, 0.5257, -0.8507],
        [0.8507, 0.0000, -0.5257],
        [0.8507, 0.0000, 0.5257],
        [-0.8507, 0.0000, -0.5257],
        [-0.8507, 0.0000, 0.5257],
    ]

    # Faces for level 0 ico-sphere
    ico_faces0 = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    verts = torch.tensor(ico_verts0, dtype=torch.float32, device=device)
    faces = torch.tensor(ico_faces0, dtype=torch.int64, device=device)

    verts *= scale
    if translation is not None:
        verts += translation

    return Meshes(verts=[verts], faces=[faces])
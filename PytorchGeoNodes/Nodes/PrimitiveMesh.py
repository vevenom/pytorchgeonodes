import torch
from pytorch3d.structures import Meshes
from PytorchGeoNodes.Nodes.PrimitiveGeometry import PrimitiveGeometry
from scipy.signal import dimpulse
from typing import List

class PrimitiveMesh(PrimitiveGeometry):
    def __init__(self, identifier, primitive_type, device=None):
        torch.nn.Module.__init__(self)
        PrimitiveGeometry.__init__(self, identifier, primitive_type, device)

        self.verts = torch.zeros((1, 0, 3), device=device)
        self.faces = torch.zeros((1, 0, 3), device=device)

        self.verts_base_primitive_ids = torch.zeros((1, 0), device=device)

        # This id changes during join functions
        self.verts_individual_primitive_ids = torch.zeros((1, 0), device=device)

    def to(self, device):
        torch.nn.Module.to(self, device)
        self.verts = self.verts.to(device)
        self.faces = self.faces.to(device)
        self.verts_base_primitive_ids = self.verts_base_primitive_ids.to(device)
        self.verts_individual_primitive_ids = self.verts_individual_primitive_ids.to(device)

    def is_empty(self):
        return self.verts.shape[1] == 0

    def set_all(self, verts, faces, base_id, individual_ids=None):
        self.verts = verts
        self.faces = faces
        self.verts_base_primitive_ids = torch.zeros_like(self.verts[..., 0]) + base_id

        if individual_ids is None:
            self.verts_individual_primitive_ids = torch.zeros_like(self.verts[..., 0])
        else:
            self.verts_individual_primitive_ids = individual_ids


    def len_verts(self):
        return self.verts.shape[1]

    def len_faces(self):
        return self.faces.shape[1]

    # @property
    # def device(self):
    #     assert False

    def get_device(self):
        return self.verts.device

    def scale(self, size):
        assert len(size.shape) == 3, "Number of dims should be 3, got {}".format(len(size.shape))
        #print("verts: ", self.verts.device)
        #print("size: ", size.device)
        self.verts = self.verts * size

    def transform(self, transform3d):
        verts = transform3d.transform_points(self.verts)

        transformed_mesh = self.clone()
        transformed_mesh.verts = verts

        return transformed_mesh

    def clone(self):
        new_primitive = PrimitiveMesh(-1, self.get_device())
        new_primitive.primitive_type = self.primitive_type

        verts = torch.zeros_like(self.verts) + self.verts
        faces = self.faces.clone()
        new_primitive.set_all(verts, faces,
                              self.verts_base_primitive_ids,
                              self.verts_individual_primitive_ids.clone())
        # new_primitive.faces = self.faces.clone()
        # new_primitive.verts_base_primitive_ids = self.verts_base_primitive_ids.clone()

        return new_primitive

    def to_mesh(self):
        return Meshes(verts=self.verts, faces=self.faces)


def join_meshes(primitive_meshes: List[PrimitiveMesh]):
    primitive_meshes = [mesh for mesh in primitive_meshes if not mesh.is_empty()] # Ignore empty primitives

    if not len(primitive_meshes):
        return PrimitiveMesh(-1, 'Empty', device=None)

    device = primitive_meshes[0].get_device()

    total_num_verts = sum(mesh.len_verts() for mesh in primitive_meshes)
    total_num_faces = sum(mesh.len_faces() for mesh in primitive_meshes)
    joined_verts = torch.zeros((1, total_num_verts, 3), device=device)
    joined_faces = torch.zeros((1, total_num_faces, 3), dtype=torch.long, device=device)
    joined_verts_base_ids = torch.zeros((1, total_num_verts), device=device)
    joined_verts_individual_ids = torch.zeros((1, total_num_verts), device=device)

    curr_num_verts = 0
    curr_num_faces = 0
    individual_ids_acc = 0
    for mesh_id, pm in enumerate(primitive_meshes):
        # if pm.is_empty():
        #     continue

        joined_verts[:, curr_num_verts:curr_num_verts + pm.len_verts()] = pm.verts
        joined_faces[:, curr_num_faces:curr_num_faces + pm.len_faces()] = pm.faces + curr_num_verts

        joined_verts_base_ids[:, curr_num_verts:curr_num_verts + pm.len_verts()] = pm.verts_base_primitive_ids

        if pm.verts_individual_primitive_ids.shape[1] > 0:
            ind_ids = pm.verts_individual_primitive_ids
            min_id = torch.min(ind_ids)
            ind_ids -= min_id
            ind_ids += individual_ids_acc
            joined_verts_individual_ids[:, curr_num_verts:curr_num_verts + pm.len_verts()] += ind_ids
            individual_ids_acc += (torch.max(ind_ids) - torch.min(ind_ids)) + 1
        else:
            joined_verts_individual_ids[:, curr_num_verts:curr_num_verts + pm.len_verts()] = individual_ids_acc
            individual_ids_acc += 1

        curr_num_verts += pm.len_verts()
        curr_num_faces += pm.len_faces()

    new_primitive = PrimitiveMesh(-1, 'Joined', device=device)
    new_primitive.set_all(joined_verts, joined_faces, joined_verts_base_ids, joined_verts_individual_ids)

    assert joined_faces.shape[0] == 1

    return new_primitive


def join_cloned_verts_faces(verts, faces, mesh, new_primitive_type):
    num_clones = verts.shape[0]
    device = verts.device

    joined_verts = torch.zeros((1, num_clones * verts.shape[1], 3), device=device)
    joined_faces = torch.zeros((1, num_clones * faces.shape[1], 3), dtype=torch.long, device=device)
    joined_verts_base_ids = torch.zeros((1, num_clones * verts.shape[1]), device=device)
    joined_verts_individual_ids = torch.zeros((1, num_clones * verts.shape[1]), device=device)

    curr_num_verts = 0
    curr_num_faces = 0
    individual_ids_acc = 0
    for clone_ind in range(num_clones):
        joined_verts[:, curr_num_verts:curr_num_verts + verts.shape[1]] = verts[clone_ind][None]
        joined_faces[:, curr_num_faces:curr_num_faces + faces.shape[1]] = faces + curr_num_verts

        joined_verts_base_ids[:, curr_num_verts:curr_num_verts + verts.shape[1]] = mesh.verts_base_primitive_ids
        if mesh.verts_individual_primitive_ids.shape[1] > 0:
            ind_ids = mesh.verts_individual_primitive_ids
            min_id = torch.min(ind_ids)
            ind_ids -= min_id
            ind_ids += individual_ids_acc
            joined_verts_individual_ids[:, curr_num_verts:curr_num_verts + mesh.len_verts()] += ind_ids
            individual_ids_acc += (torch.max(ind_ids) - torch.min(ind_ids)) + 1
        else:
            joined_verts_individual_ids[:, curr_num_verts:curr_num_verts + mesh.len_verts()] = individual_ids_acc
            individual_ids_acc += 1

        curr_num_verts += verts.shape[1]
        curr_num_faces += faces.shape[1]

    new_primitive = PrimitiveMesh(-1, new_primitive_type, device=device)
    new_primitive.set_all(joined_verts, joined_faces, joined_verts_base_ids, joined_verts_individual_ids)
    # new_primitive.verts = joined_verts
    # new_primitive.faces = joined_faces

    return new_primitive

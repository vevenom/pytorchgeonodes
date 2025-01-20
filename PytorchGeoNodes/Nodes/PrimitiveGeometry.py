import torch
from pytorch3d.structures import Meshes
from scipy.signal import dimpulse
from typing import List

class PrimitiveGeometry(torch.nn.Module):
    def __init__(self, identifier, primitive_type, device=None):
        torch.nn.Module.__init__(self)
        self.identifier = identifier
        self.primitive_type = primitive_type

    def is_empty(self):
        return True

    @staticmethod
    def create_empty(device):
        return PrimitiveGeometry(-1, 'Empty', device=device)

    def get_device(self):
        raise NotImplementedError

    def scale(self, size):
        raise NotImplementedError

    def transform(self, transform3d):
        raise NotImplementedError

    def clone(self):
        raise NotImplementedError
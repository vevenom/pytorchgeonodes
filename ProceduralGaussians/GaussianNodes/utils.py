import torch


# https://github.com/nerfstudio-project/gsplat/blob/main/examples/utils.py
def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def get_triangle_bary_coords(degree):
    if degree == 4:
        return torch.tensor(
            [[1 / 3, 1 / 3, 1 / 3],
             [2 / 3, 1 / 6, 1 / 6],
             [1 / 6, 2 / 3, 1 / 6],
             [1 / 6, 1 / 6, 2 / 3]],
            dtype=torch.float32,
            # device=self.primitive_mesh.device,
        )
    elif degree == 7:
        return torch.tensor(
            [
                [1 / 3, 1 / 3, 1 / 3],
                [2 / 3, 1 / 6, 1 / 6],
                [1 / 6, 2 / 3, 1 / 6],
                [1 / 6, 1 / 6, 2 / 3],
                [2 / 5, 2 / 5, 1 / 5],
                [2 / 5, 1 / 5, 2 / 5],
                [1 / 5, 2 / 5, 2 / 5],
            ],
            dtype=torch.float32,
            # device=self.primitive_mesh.device,
        )
    elif degree == 10:
        return torch.tensor(
            [
                [1 / 3, 1 / 3, 1 / 3],
                [2 / 3, 1 / 6, 1 / 6],
                [1 / 6, 2 / 3, 1 / 6],
                [1 / 6, 1 / 6, 2 / 3],
                [2 / 5, 2 / 5, 1 / 5],
                [2 / 5, 1 / 5, 2 / 5],
                [1 / 5, 2 / 5, 2 / 5],
                [4 / 10, 4 / 10, 2 / 10],
                [4 / 10, 2 / 10, 4 / 10],
                [2 / 10, 4 / 10, 4 / 10],
            ],
            dtype=torch.float32,
            # device=self.primitive_mesh.device,
        )
    else:
        raise NotImplementedError
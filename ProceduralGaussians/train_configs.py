# Some parts of this code are based on shape-of-motion:
# https://github.com/vye16/shape-of-motion (Accessed Jun 18th 2025)
#
# The original license applies.
from dataclasses import dataclass

@dataclass
class FGLRConfig:
    means_offsets: float = 1e-3
    opacities_offsets: float = 1e-3
    scales_offsets: float = 1e-3
    scale_thickness_offsets: float = 1e-3
    complex_offsets: float = 1e-3
    colors_offsets: float = 1e-3
    verts_offsets: float = 1e-3
    verts_colors_offsets: float = 1e-3
    verts_offsets_mlp: float = 1e-3
    means_offsets_mlp: float = 1e-3

@dataclass
class BGLRConfig:
    means: float = 1e-3
    opacities: float = 1e-3
    scales: float = 1e-3
    quats: float = 1e-3
    colors: float = 1e-3

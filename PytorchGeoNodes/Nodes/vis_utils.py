from PytorchGeoNodes.Nodes.node_types import *

# Assign unique color to each node type
class GeometryNodesVis:
    node_type_colors = {
        NodeTypes.MATH_str: (0., 0., 0.8),
        NodeTypes.FRAME_str: (0.5, 0., 0.),
        NodeTypes.VECT_MATH_str: (0., 0.8, 0.),
        NodeTypes.COMBXYZ_str: (0.5, 0., 0.5),
        NodeTypes.GROUP_INPUT_str: (0.5, 0.5, 0.),
        NodeTypes.GROUP_OUTPUT_str: (0., 0.5, 0.5),
        NodeTypes.POINTS_str: (0.5, 0.1, 0.9),
        NodeTypes.MESH_PRIMITIVE_LINE_str: (0.9, 0.1, 0.5),
        NodeTypes.MESH_PRIMITIVE_CUBE_str: (0.9, 0.5, 0.1),
        NodeTypes.INSTANCE_ON_POINTS_str: (0.1, 0.9, 0.5),
        NodeTypes.REALIZE_INSTANCES_str: (0.1, 0.5, 0.9),
        NodeTypes.GROUP_str: (0.5, 0.9, 0.1),
        NodeTypes.SWITCH_str: (0.3, 0.4, 0.7),
        NodeTypes.JOIN_GEOMETRY_str: (0.7, 0.3, 0.4)
    }

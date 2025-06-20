from pytorch3d.transforms import Transform3d
from pytorch3d.transforms import Translate
from pytorch3d.structures import Meshes


def forward_obj_gaussians(geometry_nodes, input_params_dict, rotation_matrix, translation_offset=None,
                          obj_center=None, transform2blender_coords=True):
    _, outputs = geometry_nodes.forward(input_params_dict, transform2blender_coords=transform2blender_coords)
    obj_gaussians = outputs[0][0][0]

    obj_mesh = Meshes(verts=obj_gaussians.primitive_mesh.verts, faces=obj_gaussians.primitive_mesh.faces)
    # print("Verts Size 1: ", obj_gaussians.primitive_mesh.verts.size())
    transform = Transform3d(device=obj_gaussians.get_device())

    if rotation_matrix is not None:
        transform_rot = Transform3d(matrix=rotation_matrix, device=rotation_matrix.device)
        assert rotation_matrix.shape[0] == 1
        transform = transform.compose(transform_rot)
    else:
        assert False, "We are doing experiments with rotations"

    bb = obj_mesh.get_bounding_boxes()  # (N, 3, 2)
    bb_center = (bb[:, :, 1] - bb[:, :, 0]) / 2 + bb[:, :, 0]
    bb_center_transformed = transform.transform_points(bb_center)

    # New
    transform = Translate(-2 * bb_center).compose(transform_rot)

    if obj_center is not None:
        translation = obj_center + bb_center_transformed
    else:
        translation = bb_center

    transform = transform.compose(Translate(translation))

    if translation_offset is not None:
        translate = Translate(translation_offset)

        transform = transform.compose(translate)

    obj_gaussians = obj_gaussians.transform(transform)


    obj_mesh = Meshes(verts=obj_gaussians.primitive_mesh.verts, faces=obj_gaussians.primitive_mesh.faces)

    return obj_gaussians, obj_mesh
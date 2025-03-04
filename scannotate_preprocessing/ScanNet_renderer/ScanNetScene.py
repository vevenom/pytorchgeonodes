class ScanNetScene(object):
    def __init__(self, scene_name, bbox3d_list, tmesh, sem_seg_3d, inst_labels, points, scene_type):
        self.scene_name = scene_name
        self.bbox3d_list = bbox3d_list
        self.scene_mesh = tmesh
        self.sem_seg_3d = sem_seg_3d
        self.inst_seg_3d = inst_labels
        self.points_3d = points
        self.scene_type = scene_type
        self.bbox3d_list_fused = None
        self.retrieval_time = None

    def add_box_3d_list_fused(self, box_3d_list):
        self.bbox3d_list_fused = box_3d_list


class Box3D(object):
    def __init__(self, center, basis, scale, box3d, cls_name, inst_seg_id, view_params, scannet_cls_label,
                 cad_annotation_source,
                 scan2cad_annotation_dict=None, is_box_height_expanded=False, is_box_width_expanded=False):
        self.center = center
        self.basis = basis
        self.scale = scale
        self.box3d = box3d

        self.cad_annotation_source = cad_annotation_source

        self.is_box_height_expanded = is_box_height_expanded
        self.is_box_width_expanded = is_box_width_expanded


        self.is_cad_manually_corrected = False
        self.is_cad_cloned = False
        self.use_2d_rgb_mask = False
        self.mean_IOU_mask_rgb_vs_3d = -1.

        self.cls_name = cls_name
        self.inst_seg_id = inst_seg_id
        self.view_params = view_params

        self.scan2cad_annotation_dict = scan2cad_annotation_dict
        self.scannet_cls_label = scannet_cls_label

        self.transform3d_list = None
        self.transform_dict_list = None
        self.obj_id_retrieval_list = None

        self.obj_id_similarity_list = None

        self.transform3d_refine_list = None
        self.transform_refine_dict_list = None
        self.obj_id_refined_list = None


        self.orientation_list = None
        self.orientation_list_deg = None
        self.obj_id_list = None
        self.loss_list = None
        self.mcss_iter_list = None

    def add_obj_id_retrieval_list(self, obj_id_list):
        self.obj_id_retrieval_list = obj_id_list

    def add_obj_id_similarity(self, obj_id_list):
        self.obj_id_similarity_list = obj_id_list

    def add_obj_id_refined(self, obj_id_list):
        self.obj_id_refined_list = obj_id_list

    def add_transform3D(self, transform3d_list, transform_dict_list):
        self.transform3d_list = transform3d_list
        self.transform_dict_list = transform_dict_list

    def add_transform3D_refined(self, transform3d_refined, transform_refine_dict_list):
        self.transform3d_refine_list = transform3d_refined
        self.transform_refine_dict_list = transform_refine_dict_list

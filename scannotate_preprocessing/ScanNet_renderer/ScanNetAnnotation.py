class ScanNetAnnotation(object):
    def __init__(self, scene_name, obj_annotation_list, inst_seg_3d, scene_type):
        self.scene_name = scene_name
        self.obj_annotation_list = obj_annotation_list
        self.inst_seg_3d = inst_seg_3d
        self.scene_type = scene_type


class ObjectAnnotation(object):
    def __init__(self, object_id, category_label, scannet_category_label, view_params, transform3d=None,
                 transform_dict=None, id_cad=None, catid_cad=None, cad_symmetry='__SYM_NONE', is_in_scan2cad=False,
                 scan2cad_annotation_dict=None):
        self.object_id = object_id
        self.category_label = category_label
        self.view_params = view_params
        self.scannet_category_label = scannet_category_label
        self.scan2cad_annotation_dict = scan2cad_annotation_dict

        self.transform3d = transform3d
        self.transform_dict = transform_dict
        self.id_cad = id_cad
        self.catid_cad = catid_cad
        self.cad_symmetry = cad_symmetry
        self.is_in_scan2cad = is_in_scan2cad

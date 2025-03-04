import json

table_var_order = [
    "Width",
    "Height",
    "Depth",
    "Top Shape",
    "Legs Type",
    "Has Cabinet Leg",
    "Has Mid Board",
    "Mid Leg X Scale",
    "Mid Leg Y Scale",
    "Legs Offset Y",
    "Legs Offset X",
    "Legs Scale Y",
    "Mid Board Z Scale",
    "Cabinet Width Scale",
    "Top Thickness",
    "Legs Scale X"]

sofa_var_order = [
    "Width",
    "Depth",
    "Height",
    "Has Back",
    "Back Height",
    "Back Depth",
    "Back Over-Width Scale",
    "Is L-Shaped",
    "Flip L Around Y",
    "L Depth",
    "L Width",
    "Has Left Arm",
    "Has Right Arm",
    "Arm Width",
    "Arm Depth",
    "Arm Height",
    # "Has Arm Legs",
    "Has Legs",
    "Leg Size",
    "Leg Height"
]

chair_var_order = [
    "Seat Width",
    "Seat Height",
    "Seat Thickness",
    "Seat Depth",
    "Has Back",
    "Backrest Scale",
    "Back Height",
    "Back Thickness",
    "Backrest Offset Scale",
    "Legs Type",
    "Legs Size",
    # "Bottom Star",
    "Bottom Size Scale",
    "Bottom Thickness",
    "Has Middle Support",
    "Middle Offset 2",
    "Middle Offset 1",
    "Middle Support Thickness",
    "Star Rotation",
    "Has Arms",
    "Arm Height",
    "Arm Depth Scale",
    "Arm Width",
    "Arm Thickness"
]

cabinet_var_order = [
    "Height",
    "Width",
    "Depth",
    "Board Thickness",
    "Has Drawers",
    "Number of Dividing Boards",
    "Dividing Board Thickness",
    "Has Back",
    "Has Legs",
    "Leg Width",
    "Leg Height",
    "Leg Depth",
]


table_param_types = {
    "Width": float,
    "Height": float,
    "Depth": float,
    "Top Shape": int,
    "Legs Type": int,
    "Has Cabinet Leg": int,
    "Has Mid Board": int,
    "Mid Leg X Scale": float,
    "Mid Leg Y Scale": float,
    "Legs Offset Y": float,
    "Legs Offset X": float,
    "Legs Scale Y": float,
    "Mid Board Z Scale": float,
    "Cabinet Width Scale": float,
    "Top Thickness": float,
    "Legs Scale X": float
}

sofa_param_types = {
    "Width": float,
    "Depth": float,
    "Height": float,
    "Has Back": int,
    "Back Height": float,
    "Back Depth": float,
    "Back Over-Width Scale": float,
    "Is L-Shaped": int,
    "Flip L Around Y": int,
    "L Depth": float,
    "L Width": float,
    "Has Left Arm": int,
    "Has Right Arm": int,
    "Arm Width": float,
    "Arm Depth": float,
    "Arm Height": float,
    "Has Arm Legs": int,
    "Has Legs": int,
    "Leg Size": float,
    "Leg Height": float,
}

chair_param_types = {
    "Seat Height": float,
    "Has Back": int,
    "Legs Type": int,
    "Has Arms": int,
    "Bottom Size Scale": float,
    "Seat Width": float,
    "Bottom Star": int,
    "Star Rotation": float,
    "Seat Thickness": float,
    "Seat Depth": float,
    "Has Middle Support": int,
    "Arm Height": float,
    "Backrest Scale": float,
    "Back Height": float,
    "Arm Depth Scale": float,
    "Backrest Offset Scale": float,
    "Middle Offset 2": float,
    "Middle Offset 1": float,
    "Legs Size": float,
    "Middle Support Thickness": float,
    "Bottom Thickness": float,
    "Back Thickness": float,
    "Arm Width": float,
    "Arm Thickness": float
}

cabinet_param_types = {
    "Height": float,
    "Width": float,
    "Depth": float,
    "Board Thickness": float,
    "Has Drawers": int,
    "Number of Dividing Boards": int,
    "Dividing Board Thickness": float,
    "Has Back": int,
    "Has Legs": int,
    "Leg Width": float,
    "Leg Height": float,
    "Leg Depth": float
}


def evaluate_table(reconstruction_path, gt_path, gt_params=None):
    with open(reconstruction_path, 'r') as f:
        reconstruction_json = json.load(f)
    reconstruction_json = reconstruction_json['input_dict']

    if gt_params is None:
        with open(gt_path, 'r') as f:
            gt_json = json.load(f)
        gt_json = gt_json['input_dict']
    else:
        gt_json = gt_params

    params_to_evaluate_dict = {
            "Width": True,
            "Height": True,
            "Depth": True,
            "Legs Type": True,
            "Mid Leg X Scale": gt_json['Has Mid Board'],
            "Mid Leg Y Scale": gt_json['Has Mid Board'],
            "Top Shape": True,
            "Legs Offset Y": gt_json['Top Shape'] == 0,
            "Legs Offset X": gt_json['Top Shape'] == 0,
            "Legs Scale Y": gt_json['Top Shape'] == 0,
            "Has Mid Board": gt_json['Top Shape'] == 0,
            "Mid Board Z Scale": gt_json['Top Shape'] == 0,
            "Has Cabinet Leg": gt_json['Top Shape'] == 0,
            "Cabinet Width Scale": gt_json['Top Shape'] == 0 and gt_json['Has Cabinet Leg'] == 1,
            "Top Thickness": True,
            "Legs Scale X": gt_json['Legs Type'] == 0
    }
    params_scores = {}
    for key in table_var_order:
        if not params_to_evaluate_dict[key]:
             continue

        reconstructed_value = reconstruction_json[key]
        gt_value = gt_json[key]

        if isinstance(gt_value, int) or isinstance(gt_value, int):
            params_scores[key] = int(reconstructed_value == gt_value)
        elif isinstance(gt_value, float):
            params_scores[key] = abs(reconstructed_value - gt_value)
        else:
            assert False, 'Type is weird for: {} type {}'.format(gt_value, type(gt_value))

    return params_scores


def evaluate_sofa(reconstruction_path, gt_path, gt_params=None):
    with open(reconstruction_path, 'r') as f:
        reconstruction_json = json.load(f)
    reconstruction_json = reconstruction_json['input_dict']

    if gt_params is None:
        with open(gt_path, 'r') as f:
            gt_json = json.load(f)
        gt_json = gt_json['input_dict']
    else:
        gt_json = gt_params
    gt_json['Has Legs'] = gt_json['Has Legs'] == 1 or gt_json['Has Arm Legs'] == 1

    params_to_evaluate_dict = {
        "Width": True,
        "Depth": True,
        "Height": True,
        "Has Back": True,
        "Back Height": gt_json['Has Back'] == 1,
        "Back Depth": gt_json['Has Back'] == 1,
        "Back Over-Width Scale": gt_json['Has Back'] == 1,
        "Is L-Shaped": True,
        "Flip L Around Y": gt_json['Is L-Shaped'] == 1,
        "L Depth": gt_json['Is L-Shaped'] == 1,
        "L Width": gt_json['Is L-Shaped'] == 1,
        "Has Left Arm": True,
        "Has Right Arm": True,
        "Arm Width": gt_json['Has Left Arm'] == 1 or gt_json['Has Right Arm'] == 1,
        "Arm Depth": gt_json['Has Left Arm'] == 1 or gt_json['Has Right Arm'] == 1,
        "Arm Height": gt_json['Has Left Arm'] == 1 or gt_json['Has Right Arm'] == 1,
        "Has Arm Legs": False, # Skip this as it was hard to manually annotate
        "Has Legs": True,
        "Leg Size": gt_json['Has Legs'] == 1,
        "Leg Height": gt_json['Has Legs'] == 1
    }
    params_scores = {}


    for key in sofa_var_order:
        if not params_to_evaluate_dict[key]:
             continue

        reconstructed_value = reconstruction_json[key]
        gt_value = gt_json[key]

        if isinstance(gt_value, int) or isinstance(gt_value, int):
            params_scores[key] = int(reconstructed_value == gt_value)
        elif isinstance(gt_value, float):
            params_scores[key] = abs(reconstructed_value - gt_value)
        else:
            assert False, 'Type is weird for: {} type {}'.format(gt_value, type(gt_value))

    return params_scores


def evaluate_chair(reconstruction_path, gt_path, gt_params=None):
    with open(reconstruction_path, 'r') as f:
        reconstruction_json = json.load(f)
    reconstruction_json = reconstruction_json['input_dict']

    if gt_params is None:
        with open(gt_path, 'r') as f:
            gt_json = json.load(f)
        gt_json = gt_json['input_dict']
    else:
        gt_json = gt_params
    gt_json['Bottom Star'] = gt_json['Bottom Star'] == 1 or gt_json['Legs Type'] == 1

    params_to_evaluate_dict = {
        "Seat Width": True,
        "Seat Height": True,
        "Seat Thickness": True,
        "Seat Depth": True,
        "Has Back": True,
        "Backrest Scale": gt_json['Has Back'] == 1,
        "Back Height": gt_json['Has Back'] == 1,
        "Back Thickness": gt_json['Has Back'] == 1,
        "Backrest Offset Scale": gt_json['Has Back'] == 1,
        "Legs Type": True,
        "Legs Size": True,
        "Bottom Star": False,
        "Bottom Size Scale": gt_json['Legs Type'] == 1,
        "Bottom Thickness": gt_json['Legs Type'] == 1,
        "Star Rotation": gt_json['Legs Type'] == 1,
        "Has Middle Support": gt_json['Legs Type'] == 0,
        "Middle Offset 2": gt_json['Legs Type'] == 0,
        "Middle Offset 1": gt_json['Legs Type'] == 0,
        "Middle Support Thickness": gt_json['Has Middle Support'] == 1 and gt_json['Legs Type'] == 0,
        "Has Arms": True,
        "Arm Height": gt_json['Has Arms'] == 1,
        "Arm Depth Scale": gt_json['Has Arms'] == 1,
        "Arm Width": gt_json['Has Arms'] == 1,
        "Arm Thickness": gt_json['Has Arms'] == 1
    }
    params_scores = {}


    for key in chair_var_order:
        if not params_to_evaluate_dict[key]:
             continue

        reconstructed_value = reconstruction_json[key]
        gt_value = gt_json[key]

        if isinstance(gt_value, int) or isinstance(gt_value, int):
            params_scores[key] = int(reconstructed_value == gt_value)
        elif isinstance(gt_value, float):
            params_scores[key] = abs(reconstructed_value - gt_value)
        else:
            assert False, 'Type is weird for: {} type {}'.format(gt_value, type(gt_value))

    return params_scores


def evaluate_cabinet(reconstruction_path, gt_path, gt_params=None):
    with open(reconstruction_path, 'r') as f:
        reconstruction_json = json.load(f)
    reconstruction_json = reconstruction_json['input_dict']

    if gt_params is None:
        with open(gt_path, 'r') as f:
            gt_json = json.load(f)
        gt_json = gt_json['input_dict']
    else:
        gt_json = gt_params

    params_to_evaluate_dict = {
        "Height": True,
        "Width": True,
        "Depth": True,
        "Board Thickness": True,
        "Has Drawers": True,
        "Number of Dividing Boards": gt_json['Has Drawers'] == 0,
        "Dividing Board Thickness": gt_json['Has Drawers'] == 0,
        "Has Back": gt_json['Has Drawers'] == 0,
        "Has Legs": True,
        "Leg Width": gt_json['Has Legs'] == 1,
        "Leg Height": gt_json['Has Legs'] == 1,
        "Leg Depth": gt_json['Has Legs'] == 1,
    }
    params_scores = {}


    for key in cabinet_var_order:
        if not params_to_evaluate_dict[key]:
             continue

        reconstructed_value = reconstruction_json[key]
        gt_value = gt_json[key]

        if isinstance(gt_value, int) or isinstance(gt_value, int):
            params_scores[key] = int(reconstructed_value == gt_value)
        elif isinstance(gt_value, float):
            params_scores[key] = abs(reconstructed_value - gt_value)
        else:
            assert False, 'Type is weird for: {} type {}'.format(gt_value, type(gt_value))

    return params_scores
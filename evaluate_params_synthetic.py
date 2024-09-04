import pickle
import os
import argparse
import json
# from misc.utils import write_json, read_json
import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score

dummy_dict = {
    'cabinet':
        {'Width': [],
         'Height': [],
         'Depth': [],
         'Board Thickness': [],
         'Has Back': [],
         'Has Legs': [],
         'Leg Width': [],
         'Leg Height': [],
         'Leg Depth': [],
         'Number of Dividing Boards': [],
         'Dividing Board Thickness': [],
         'Has Drawers': [],
         'rotation_y': []},
    'chair':
        {'Legs Type': [],
         'Legs Size': [],
         'Has Middle Support': [],
         'Middle Offset': [],
         'Bottom Thickness': [],
         'Bottom Size Scale': [],
         'Seat Height': [],
         'Seat Width': [],
         'Seat Depth': [],
         'Seat Thickness': [],
         'Has Back': [],
         'Back Height': [],
         'Backrest Scale': [],
         'Back Thickness': [],
         'Backrest Offset Scale': [],
         'Has Arms': [],
         'Arm Depth Scale': [],
         'Arm Height': [],
         'Arm Width': [],
         'Arm Thickness': []
         },
    'sofa':
        {'Width': [],
         'Height': [],
         'Depth': [],
         'Has Legs': [],
         'Leg Size': [],
         'Has Left Arm': [],
         'Has Right Arm': [],
         'Arm Width': [],
         'Arm Height': [],
         'Arm Depth': [],
         'Has Arm Legs': [],
         'Has Back': [],
         'Back Height': [],
         'Back Depth': [],
         'Is L-Shaped': [],
         'L Width': [],
         'L Depth': [],
         'Flip L Around Y': [],
         # 'rotation_y': []
         },
    'table':
        {'Width': [],
         'Height': [],
         'Depth': [],
         'Top Shape': [],
         'Top Thickness': [],
         'Legs Type': [],
         'Mid Leg X Scale': [],
         'Mid Leg Y Scale': [],
         'Has Mid Board': [],
         'Mid Board Z Scale': [],
         }
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Reconstruct scannotate objects')
    parser.add_argument('--category', type=str, default='cabinet', help='Object category')
    parser.add_argument('--experiments_path', type=str, help='Experiment path')
    parser.add_argument('--experiment_name', type=str, help='Experiment name')
    parser.add_argument('--dataset_path', type=str, help='Dataset path')
    args = parser.parse_args()

    object_category = args.category
    experiments_path = args.experiments_path
    experiment_name = args.experiment_name
    dataset_path = args.dataset_path

    dummy_dict = dummy_dict[object_category]
    dummy_dict['rotation_y'] = []

    # Rotation dummy list if we assume that object can have sym level 4 ( 90 degree symmetry)
    rot_dummy_list = np.asarray([-270, -180, -90, 0, 90, 180, 270])

    sp_pred_dict = deepcopy(dummy_dict)
    sp_gt_dict = deepcopy(dummy_dict)
    evaluation_metrics_dict = deepcopy(dummy_dict)

    # TODO decide here which experiment folder to evaluate, and if evaluation results should be normalized
    # ['synthetic_reconstructions','synthetic_reconstructions_no_refinement','synthetic_reconstructions_no_refinement_no_exploit']

    gt_dir = os.path.join(dataset_path, object_category)
    recon_dir = os.path.join(
        experiments_path,
        'reconstructed_meshes',
        experiment_name,
        object_category)

    eval_output_folder = os.path.join(experiments_path, 'evaluation_results')
    eval_output_path = os.path.join(eval_output_folder, object_category + '_' + experiment_name + '.json')
    os.makedirs(eval_output_folder, exist_ok=True)

    folder_list = os.listdir(recon_dir)

    for folder in folder_list:

        gt_anno_path = os.path.join(gt_dir, folder, 'scene_dict.pkl')
        recon_anno_path = os.path.join(recon_dir, folder, 'final_solution.json')

        if not os.path.exists(gt_anno_path) or not os.path.exists(recon_anno_path):
            print(('Annotations or reconstruction not available for file {}'.format(gt_anno_path)))
            continue

        # Load scannotate scene pickle
        with open(gt_anno_path, 'rb') as f:
            scene_pkl = pickle.load(f)

        with open(recon_anno_path, 'r') as f:
            predictions_json = json.load(f)
        # predictions_json = read_json(recon_anno_path)

        # Iterate through all examples and add ground truth and predictions to dicts for further processing
        for key in dummy_dict:

            if key == 'rotation_y':
                continue

            if object_category == 'chair':
                # TODO check dependencies here and skip accordingly
                if key in ['Has Middle Support', 'Middle Offset']:
                    if str(scene_pkl['sp_params']['Legs Type']) != '0':
                        continue

                if key in ['Middle Offset']:
                    if scene_pkl['sp_params']['Has Middle Support'] == False:
                        continue

                if key in ['Bottom Thickness', 'Bottom Size Scale']:
                    if str(scene_pkl['sp_params']['Legs Type']) != '1':
                        continue

                if key in ['Back Height', 'Backrest Scale', 'Back Thickness', 'Backrest Offset Scale']:
                    if scene_pkl['sp_params']['Has Back'] == False:
                        continue

                if key in ['Arm Depth Scale', 'Arm Height', 'Arm Width', 'Arm Thickness']:
                    if scene_pkl['sp_params']['Has Arms'] == False:
                        continue
            elif object_category == 'cabinet':
                if key in ['Leg Width', 'Leg Height', 'Leg Depth']:
                    if scene_pkl['sp_params']['Has Legs'] == False:
                        continue

                if key in ['Number of Dividing Boards', 'Dividing Board Thickness', 'Has Back']:
                    if scene_pkl['sp_params']['Has Drawers'] == True:
                        continue
            elif object_category == 'sofa':
                if key in ['Leg Size']:
                    if scene_pkl['sp_params']['Has Legs'] == False:
                        continue

                if key in ['Arm Width', 'Arm Height', 'Arm Depth', 'Has Arm Legs']:
                    if scene_pkl['sp_params']['Has Left Arm'] == False and scene_pkl['sp_params'][
                        'Has Right Arm'] == False:
                        continue

                if key in ['Back Height', 'Back Depth']:
                    if scene_pkl['sp_params']['Has Back'] == False:
                        continue

                if key in ['Flip L Around Y', 'L Depth', 'L Width']:
                    if scene_pkl['sp_params']['Is L-Shaped'] == False:
                        continue
            elif object_category == 'table':
                if key in ['Mid Leg X Scale', 'Mid Leg Y Scale']:
                    if str(scene_pkl['sp_params']['Legs Type']) != '0':
                        continue

                if key in ['Has Mid Board']:
                    if str(scene_pkl['sp_params']['Legs Type']) != '2':
                        continue

                if key in ['Mid Board Z Scale']:
                    if str(scene_pkl['sp_params']['Has Mid Board']) == False:
                        continue

            sp_pred_dict[key].append(scene_pkl['sp_params'][key])
            sp_gt_dict[key].append(predictions_json['input_dict'][key])

            if key in ['Legs Type']:
                sp_pred_dict[key].append(int(scene_pkl['sp_params'][key]))
                sp_gt_dict[key].append(int(predictions_json['input_dict'][key]))
            else:
                sp_pred_dict[key].append(scene_pkl['sp_params'][key])
                sp_gt_dict[key].append(predictions_json['input_dict'][key])

        pred_rot = np.rad2deg(predictions_json['rotation_angle_y'])
        gt_rot = np.rad2deg(scene_pkl['rotation_y'].item())
        if pred_rot < 0.:
            pred_rot += 360.

        pred_rot_sym = pred_rot + rot_dummy_list
        pred_rot_diff = pred_rot_sym - gt_rot
        pred_rot_final = pred_rot_sym[np.argmin(np.abs(pred_rot_diff))]

        sp_pred_dict['rotation_y'].append(gt_rot)
        sp_gt_dict['rotation_y'].append(pred_rot_final)

    # Calculate metrics
    for key in evaluation_metrics_dict:
        # For continuous parameters, calculate mean absolute distance
        if isinstance(sp_gt_dict[key][0], float) and key != 'rotation_y':
            print(key)
            print(sp_gt_dict[key])
            print(sp_pred_dict[key])
            distance = np.mean(np.abs(np.asarray(sp_gt_dict[key]) - np.asarray(sp_pred_dict[key])))
            evaluation_metrics_dict[key] = distance * 100

        elif key == 'rotation_y':
            distance = np.mean(np.abs(np.asarray(sp_gt_dict[key]) - np.asarray(sp_pred_dict[key])))
            evaluation_metrics_dict[key] = distance

        # For boolean parameters, calculate Precision
        else:
            print(key)
            print(sp_gt_dict[key])

            predictions = sp_gt_dict[key]
            gt_values = sp_pred_dict[key]

            acc_score = accuracy_score(y_true=gt_values, y_pred=predictions)
            acc_score = np.round(acc_score, decimals=3)

            evaluation_metrics_dict[key] = acc_score * 100

    print(evaluation_metrics_dict)
    with open(eval_output_path, 'w') as f:
        json.dump(evaluation_metrics_dict, f)

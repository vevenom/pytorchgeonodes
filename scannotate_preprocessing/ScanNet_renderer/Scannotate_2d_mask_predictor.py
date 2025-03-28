# coding: utf-8
import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import argparse
from shapely.geometry import Polygon
from skimage.morphology import medial_axis, skeletonize
import os
from config import load_config
import pickle
from utils_CAD_retrieval import load_depth_img
from utils import Rz
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from pgn_utils import *


parser = argparse.ArgumentParser(description="Object Mask Prediction for ScanNet")
parser.add_argument("--config", type=str,
                    default=os.path.join(parent, 'config/0_Scannotate_masks.ini'),
                    help="Path to configuration file")

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def calc_valid_masks_for_2d(inst_seg, R, T, depth_path, label, img_scale, box_final_, intri, rgb_im_np,
                            inst_seg_folder, frame_cnt, cnt, target_height, target_width):
    mask = np.zeros_like(inst_seg)
    mask_vis = np.zeros_like(inst_seg)

    mask[inst_seg == label] = 1.
    mask_vis[inst_seg == label] = 255.

    # path_tmp = os.path.join('/home/stefan/PycharmProjects/Scannet_Total3D/demo_output/ScanNet_prepro/view_sel')
    # cv2.imwrite(os.path.join(path_tmp, 'test.png'), mask_vis)

    R = np.squeeze(R, 0)
    rot_tmp3 = Rz(np.deg2rad(180))
    T2 = np.eye(4)
    T2[:3, :3] = rot_tmp3

    M = np.eye(4)
    M[0:3, 0:3] = R.T  # because of py3d convention
    M[0:3, 3] = T
    M = np.dot(T2, M)

    x_coord_list = []
    y_coord_list = []
    min_depth = np.inf
    max_depth = -1.
    for point_ in box_final_:
        new_point = np.ones((4, 1))
        new_point[0:3, 0] = point_
        v_cam = np.dot(M, new_point)

        depth = v_cam[2]
        if depth > max_depth:
            max_depth = depth
        if depth < min_depth:
            min_depth = depth
        point = v_cam[0:3]

        if point[2] <= 0:
            point[2] = 0.0001

        point = point / point[2][None]
        u = intri[0, 0] * point[0] + intri[0, 2]
        v = intri[1, 1] * point[1] + intri[1, 2]
        pixels = np.array((int(v), int(u)))
        x_coord_list.append(pixels[0])
        y_coord_list.append(pixels[1])

        if pixels[0] < 0 or pixels[0] > target_height:
            continue
        if pixels[1] < 0 or pixels[1] > target_width:
            continue

        rgb_im_np[pixels[0] - 3:pixels[0] + 3, pixels[1] - 3:pixels[1] + 3, :] = (0, 1, 0)

        # path_tmp = os.path.join(
        #    '/home/stefan/PycharmProjects/Scannet_Total3D/demo_output/ScanNet_prepro/view_sel')
        # cv2.imwrite(os.path.join(path_tmp, 'box_test.png'), rgb_im_np * 255)

    depth_im = load_depth_img(depth_path)
    valid_depth_mask = np.ones_like(depth_im)
    valid_depth_mask[depth_im == 0] = 0

    valid_depth_mask[depth_im > max_depth] = 0
    valid_depth_mask[depth_im < min_depth] = 0

    depth_unknown_mask = np.zeros_like(valid_depth_mask)
    depth_unknown_mask[depth_im == 0] = 1
    depth_unknown_mask_vis = np.copy(rgb_im_np)
    depth_unknown_mask_vis[depth_unknown_mask == 1] = (0, 125, 255)

    v_depth_img = np.zeros_like(rgb_im_np)
    v_depth_img[valid_depth_mask == 1.] = (0, 255, 255)

    y = np.asarray(x_coord_list)
    x = np.asarray(y_coord_list)

    # x = np.asarray(x_coord_list)
    # y = np.asarray(y_coord_list)

    # x[np.where(x < 0)] =0
    # y[np.where(y < 0)] =0
    # x[np.where(x > (rgb_im_np.shape[0] - 1))] =(rgb_im_np.shape[0] - 1)
    # y[np.where(y > (rgb_im_np.shape[1] - 1))] =(rgb_im_np.shape[1] - 1)

    # image2d = Image.fromarray(np.uint8(rgb_im_np*255))
    # #image2D = rgb_im_np.convert('RGB')
    # draw2D = ImageDraw.Draw(image2d)
    # draw2D.polygon([(x[0], y[0]), (x[1], y[1]), (x[2], y[2]), (x[3], y[3])], outline=1, fill=255)
    # draw2D.polygon([(x[0], y[0]), (x[1], y[1]), (x[5], y[5]), (x[4], y[4])], outline=1, fill=255)
    # draw2D.polygon([(x[2], y[2]), (x[3], y[3]), (x[7], y[7]), (x[6], y[6])], outline=1, fill=255)
    # draw2D.polygon([(x[4], y[4]), (x[5], y[5]), (x[6], y[6]), (x[7], y[7])], outline=1, fill=255)
    # draw2D.polygon([(x[0], y[0]), (x[3], y[3]), (x[7], y[7]), (x[4], y[4])], outline=1, fill=255)
    # draw2D.polygon([(x[1], y[1]), (x[2], y[2]), (x[6], y[6]), (x[5], y[5])], outline=1, fill=255)
    #
    # test = np.asarray(draw2D.im)
    # test = np.reshape(test, (480,640,-1))
    #
    # cv2.imwrite(os.path.join(inst_seg_folder, 'valid_pix_' + str(cnt) + '_' + str(frame_cnt) + '.jpg'), test)

    overlay = rgb_im_np.copy() * 0
    polygon = Polygon([(x[0], y[0]), (x[1], y[1]), (x[2], y[2]), (x[3], y[3])])
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exterior = [int_coords(polygon.exterior.coords)]
    cv2.fillPoly(overlay, exterior, color=(255, 255, 0))

    polygon = Polygon([(x[0], y[0]), (x[1], y[1]), (x[5], y[5]), (x[4], y[4])])
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exterior = [int_coords(polygon.exterior.coords)]
    cv2.fillPoly(overlay, exterior, color=(255, 255, 0))

    polygon = Polygon([(x[2], y[2]), (x[3], y[3]), (x[7], y[7]), (x[6], y[6])])
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exterior = [int_coords(polygon.exterior.coords)]
    cv2.fillPoly(overlay, exterior, color=(255, 255, 0))

    polygon = Polygon([(x[4], y[4]), (x[5], y[5]), (x[6], y[6]), (x[7], y[7])])
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exterior = [int_coords(polygon.exterior.coords)]
    cv2.fillPoly(overlay, exterior, color=(255, 255, 0))

    polygon = Polygon([(x[0], y[0]), (x[3], y[3]), (x[7], y[7]), (x[4], y[4])])
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exterior = [int_coords(polygon.exterior.coords)]
    cv2.fillPoly(overlay, exterior, color=(255, 255, 0))

    polygon = Polygon([(x[1], y[1]), (x[2], y[2]), (x[6], y[6]), (x[5], y[5])])
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exterior = [int_coords(polygon.exterior.coords)]
    cv2.fillPoly(overlay, exterior, color=(255, 255, 0))

    img_mask_pred = ((rgb_im_np * 255.) * 0.5 + (overlay) * 0.5)

    # path_tmp = os.path.join(
    #     '/home/stefan/PycharmProjects/Scannet_Total3D/demo_output/ScanNet_prepro/view_sel')
    # cv2.imwrite(os.path.join(path_tmp, 'overlay.png'), img_mask_pred)

    img_mask_pred_2 = ((rgb_im_np * 255.) * 0.5 + (v_depth_img) * 0.5)
    # path_tmp = os.path.join(
    #     '/home/stefan/PycharmProjects/Scannet_Total3D/demo_output/ScanNet_prepro/view_sel')
    # cv2.imwrite(os.path.join(path_tmp, 'valid_depth.png'), img_mask_pred_2)

    v_final = np.zeros_like(rgb_im_np)
    valid_xy = np.zeros_like(valid_depth_mask)
    valid_xy[overlay[:, :, 0] == 255] = 1
    final_map = np.logical_and(overlay[:, :, 0], valid_depth_mask)
    v_final[final_map] = (255, 0, 255)

    img_mask_pred_3 = ((rgb_im_np * 255.) * 0.5 + (v_final) * 0.5)
    # path_tmp = os.path.join(
    #     '/home/stefan/PycharmProjects/Scannet_Total3D/demo_output/ScanNet_prepro/view_sel')
    # cv2.imwrite(os.path.join(path_tmp, 'valid_final.png'), img_mask_pred_3)

    img_mask_pred_4 = ((rgb_im_np * 255.) * 0.5 + (depth_unknown_mask_vis) * 0.5)

    pred_im_out = np.hstack((img_mask_pred, img_mask_pred_2, img_mask_pred_3, img_mask_pred_4))

    # Visualization of maps!!
    # cv2.imwrite(os.path.join(inst_seg_folder, 'valid_pix_' + str(cnt) + '_' + str(frame_cnt) + '.jpg'), pred_im_out)

    # final_map[overlay[:, :, 0] == 0] = -1

    #valid_z, valid_xy, valid_xyz, depth_unknown_mask, min_depth, valid_maps_stacked = calc_valid_masks_for_2d()
    overlay_mask = np.zeros_like(valid_xy)
    overlay_mask[overlay[:, :, 0] == 255] = 1
    return valid_depth_mask, valid_xy, final_map, depth_unknown_mask, min_depth, pred_im_out, overlay_mask


def calc_2d_inst_seg(predictor, valid_z, valid_xy, n_clicks, eval_ritm, device, mode, im_seg_thresh, target_iou,
                     inst_seg, label, rgb_im_np, img_rgb, valid_xyz,
                     skel, distance, inst_seg_2d_folder, frame_cnt,
                     cnt, mseg_labels, mask_other_instances, depth_unknown_mask, target_height, target_width):
    img_mask_pred_list = []
    img_mask_gt_list = []
    mask = np.zeros_like(inst_seg)
    mask_vis = np.zeros_like(inst_seg)

    mask[inst_seg == label] = 1.
    mask_vis[inst_seg == label] = 255.

    # todo change mask_gt
    mask_gt = mask

    img_rgb = img_rgb.float()
    _, sample_ious, _, inst_seg_2d_predicted = evaluate_sample(img_rgb, mask_gt, predictor, target_iou,
                                                               skel, distance, valid_xyz, inst_seg_2d_folder,
                                                               frame_cnt, cnt, mseg_labels, mask_other_instances,
                                                               depth_unknown_mask,
                                                               valid_xy, pred_thr=im_seg_thresh, min_clicks=1,
                                                               max_clicks=n_clicks,
                                                               sample_id=None, callback=None)
    # img_mask_gt_list.append(img_mask_gt)
    # img_mask_pred_list.append(img_mask_pred)

    return inst_seg_2d_predicted


def calc_skeleton(mask, filename1, border_distance):
    # close open mask to clean up small regions and make 3 channels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # erode mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    erode = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)

    erode = np.pad(erode, 1, mode='constant')

    # Compute the medial axis (skeleton) and the distance transform

    skel, distance = medial_axis(erode, return_distance=True)
    skel = skel[1:-1, 1:-1]
    distance = distance[1:-1, 1:-1]

    points_all = np.where(distance > 2.)
    points_all_ = np.logical_and(skel, distance > border_distance)
    # # Compare with other skeletonization algorithms
    skeleton = skeletonize(erode)
    skeleton = skeleton[1:-1, 1:-1]

    skeleton_lee = skeletonize(erode, method='lee')

    # Distance to the background for pixels of the skeleton
    dist_on_skel = distance * skel

    return skeleton, distance, points_all_


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def mask_with_prompt(image, sam, input_point, input_label):
    #
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # plt.axis('on')
    # plt.show()

    predictor = SamPredictor(sam)

    predictor.set_image(image)

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.show()

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # print(masks.shape)  # (number_of_masks) x H x W
    #
    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(image)
    #     show_mask(mask, plt.gca())
    #     show_points(input_point, input_label, plt.gca())
    #     plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.show()

    return masks, scores, logits


def all_masks_generation(image, sam):
    # mask_generator = SamAutomaticMaskGenerator(model=sam,pred_iou_thresh=0.8)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    masks = mask_generator.generate(image)
    print(len(masks))
    print(masks[0].keys())

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()
    pass


def calc_all_boxes_2d(inst_seg, R, T, depth_path, label, img_scale,
                      # box_final_,
                      intri, rgb_im_np, inst_seg_folder, frame_cnt, cnt, target_height, target_width, frame_id,
                      box_dict_3d):
    box_dict_2d = {}
    all_box2d_list_from_3dbox = []
    all_box2d_list_from_3dmask = []

    # instances = np.unique(inst_seg)
    instances = list(box_dict_3d.keys())

    for instance in instances:
        if instance >= 200 or instance < 0:
            continue
        inst_map = np.zeros_like(inst_seg)

        inst_map[inst_seg == instance] = 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inst_map = cv2.morphologyEx(inst_map, cv2.MORPH_OPEN, kernel)
        inst_map = cv2.morphologyEx(inst_map, cv2.MORPH_CLOSE, kernel)

        # erode mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inst_map = cv2.morphologyEx(inst_map, cv2.MORPH_ERODE, kernel)

        if np.sum(inst_map) < 400 and label != instance:
            continue

        # calc center of instance map
        skel, distance, points_all_ = calc_skeleton(inst_map, 'test', border_distance=10)
        max_point = np.where(distance == np.max(distance))
        input_point_tmp = np.zeros([1, 2])
        input_point_tmp[:, 0] = max_point[1][0]
        input_point_tmp[:, 1] = max_point[0][0]

        y, x = np.where(inst_map)
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        ## Draw a diagonal blue line with thickness of 5 px
        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        box_2d = np.array([x_min, y_min, x_max, y_max])
        box_dict_tmp = {}
        box_dict_tmp['box2d_from_mask3d'] = box_2d
        box_dict_tmp['point_center_from_mask3d'] = input_point_tmp

        box_dict_2d[instance] = box_dict_tmp
        all_box2d_list_from_3dmask.append(box_2d)

    R = np.squeeze(R, 0)
    rot_tmp3 = Rz(np.deg2rad(180))
    T2 = np.eye(4)
    T2[:3, :3] = rot_tmp3

    M = np.eye(4)
    M[0:3, 0:3] = R.T  # because of py3d convention
    M[0:3, 3] = T
    M = np.dot(T2, M)
    for inst_label, box_dict_tmp in box_dict_2d.items():

        box_final_ = box_dict_3d[int(inst_label)]
        x_coord_list = []
        y_coord_list = []
        min_depth = np.inf
        max_depth = -1.
        for point_ in box_final_:
            new_point = np.ones((4, 1))
            new_point[0:3, 0] = point_
            v_cam = np.dot(M, new_point)

            depth = v_cam[2]
            if depth > max_depth:
                max_depth = depth
            if depth < min_depth:
                min_depth = depth
            point = v_cam[0:3]

            if point[2] <= 0:
                point[2] = 0.0001

            point = point / point[2][None]
            u = intri[0, 0] * point[0] + intri[0, 2]
            v = intri[1, 1] * point[1] + intri[1, 2]
            pixels = np.array((int(v), int(u)))
            x_coord_list.append(pixels[0])
            y_coord_list.append(pixels[1])

        y = np.asarray(x_coord_list)
        x = np.asarray(y_coord_list)

        x[x < 0] = 0
        y[y < 0] = 0
        x[x >= target_width] = target_width - 1
        y[y >= target_height] = target_height - 1

        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        ## Draw a diagonal blue line with thickness of 5 px
        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        box_2d = np.array([x_min, y_min, x_max, y_max])

        box_dict_tmp['box2d_from_box3d'] = box_2d
        all_box2d_list_from_3dbox.append(box_2d)

        # path_tmp = os.path.join(
        #    '/home/stefan/PycharmProjects/segment-anything/debug_output')
        # cv2.imwrite(os.path.join(path_tmp, 'box_test.png'), img)

    return box_dict_2d, all_box2d_list_from_3dbox, all_box2d_list_from_3dmask


def vis_sam_masks(sam_masks_list, rgb_im_np_tmp):
    pred_img = None

    if len(sam_masks_list) == 0:
        rgb_im_copy = np.copy(rgb_im_np_tmp)
        inst_seg_2d_predicted = np.zeros((rgb_im_np_tmp.shape[0], rgb_im_np_tmp.shape[1]))  # np.copy(mask_)
        v_depth_img = np.zeros_like(rgb_im_copy)
        v_depth_img[inst_seg_2d_predicted == 1.] = (255, 0, 0)
        pred_img = ((rgb_im_copy * 255.) * 0.5 + (v_depth_img) * 0.5)

    else:
        for mask_ in sam_masks_list:
            rgb_im_copy = np.copy(rgb_im_np_tmp)
            inst_seg_2d_predicted = np.copy(mask_)
            v_depth_img = np.zeros_like(rgb_im_copy)
            v_depth_img[inst_seg_2d_predicted == 1.] = (255, 0, 0)
            img_mask_pred = ((rgb_im_copy * 255.) * 0.5 + (v_depth_img) * 0.5)

            if pred_img is None:
                pred_img = img_mask_pred
            else:
                pred_img = np.hstack((pred_img, img_mask_pred))

    return pred_img


def calc_masks_with_multiple_boxes(sam_masks_list, sam_score_list, box_dict_2d, predictor, image, sam_score_thres,
                                   label):
    target_box = box_dict_2d[int(label)]['box2d_from_box3d']
    all_boxes_list = []
    all_boxes_list.append(target_box)
    for id, box2d in box_dict_2d.items():
        if id == int(label):
            continue
        else:
            all_boxes_list.append(box2d['box2d_from_mask3d'])

    allbox_arry = np.asarray(all_boxes_list)
    input_boxes = torch.from_numpy(allbox_arry).to(device)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    masks, iou_predictions, low_res_masks = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=True,
    )

    for i in range(masks.shape[1]):
        mask_out = None
        for mask_cnt, (mask_tensor, score_tensor) in enumerate(zip(masks[:, i, :, :], iou_predictions[:, i])):
            if mask_cnt == 0:
                mask_out = mask_tensor.cpu().numpy()
                score_out = score_tensor.cpu().numpy()
        else:
            mask_out = np.logical_and(mask_out, np.logical_not(mask_tensor.cpu().numpy()))
        # if iou_predictions[0, i] > sam_score_thres:
        sam_masks_list.append(mask_out)
        sam_score_list.append(score_out.item())

    return sam_masks_list, sam_score_list


def calc_masks_with_center_point(sam_masks_list, sam_score_list, predictor, sam_score_thres, box_dict_2d, label, mask,
                                 num_pos_points=2):
    all_points_list = []
    all_labels_list = []
    mask_tmp = np.copy(mask)
    for id, box2d in box_dict_2d.items():
        if id == int(label):
            point = box2d['point_center_from_mask3d']
            all_points_list.append(point)
            all_labels_list.append(1)
            point_ = np.copy(point)
            for i in range(num_pos_points):
                mask_tmp[int(point_[0, 1]), int(point_[0, 0])] = 0

                mask_tmp = np.pad(mask_tmp, 1, mode='constant')
                _, distance = medial_axis(mask_tmp, return_distance=True)
                distance = distance[1:-1, 1:-1]

                max_point = np.where(distance == np.max(distance))
                input_points_tmp = np.zeros([1, 2])
                input_points_tmp[:, 0] = max_point[1][0]
                input_points_tmp[:, 1] = max_point[0][0]

                all_points_list.append(input_points_tmp)
                all_labels_list.append(1)

                point_ = input_points_tmp

    input_points_all_ary = np.asarray(all_points_list)
    input_points_all_ary = np.squeeze(input_points_all_ary, axis=1)
    input_lables_all_ary = np.asarray(all_labels_list)

    #    input_points = box_dict_2d[int(label)]['point_center_from_mask3d']
    #    input_labels = np.ones([input_points.shape[0]])

    masks, scores, logits = predictor.predict(
        point_coords=input_points_all_ary,
        point_labels=input_lables_all_ary,
        # box=input_box,
        multimask_output=True,
    )
    for score, mask_ in zip(scores, masks):
        # if score > sam_score_thres:
        sam_masks_list.append(mask_)
        sam_score_list.append(score)

    return sam_masks_list, sam_score_list


def calc_masks_with_pos_neg_points(sam_masks_list, predictor, valid_xyz, depth_unknown_mask, label,
                                   sam_score_thres, mask, box_dict_2d):
    valid_xyz_and_unknown_depth = np.logical_or(valid_xyz, depth_unknown_mask)
    mask_target_not = np.zeros_like(mask)
    mask_target_not[mask == 0] = 1

    mask_not = np.logical_and(mask_target_not, np.logical_not(valid_xyz_and_unknown_depth))

    input_box = box_dict_2d[int(label)]['box2d_from_box3d']
    mask_big = np.zeros_like(mask_not)
    mask_big[input_box[1]:input_box[3], input_box[0]:input_box[2]] = 1

    mask_not_big = np.logical_and(mask_not, mask_big)
    mask_not_big = np.pad(mask_not_big, 1, mode='constant')
    skel = medial_axis(mask_not_big, return_distance=False)
    skel = skel[1:-1, 1:-1]

    invalid_points_coordinates = np.where(skel == 1)
    invalid_points_coordinates = np.asarray(invalid_points_coordinates).T

    max_points = 8
    if invalid_points_coordinates.shape[0] > max_points:
        num_points = max_points
    else:
        num_points = invalid_points_coordinates.shape[0]

    point_select = np.linspace(0, invalid_points_coordinates.shape[0] - 1, num_points).astype(int)
    valid_points_coordinates_selected = invalid_points_coordinates[point_select]
    input_points_neg = np.zeros_like(valid_points_coordinates_selected)
    input_points_neg[:, 0] = invalid_points_coordinates[point_select, 1]
    input_points_neg[:, 1] = invalid_points_coordinates[point_select, 0]

    input_labels_neg = np.zeros([input_points_neg.shape[0]])

    # max_point = np.where(distance == np.max(distance))
    # input_points = np.zeros([1, 2])
    # input_points[:, 0] = max_point[1][0]
    # input_points[:, 1] = max_point[0][0]
    # input_labels = np.ones([input_points.shape[0]])

    input_points = box_dict_2d[int(label)]['point_center_from_mask3d']
    input_labels = np.ones([input_points.shape[0]])

    input_points_all = np.concatenate((input_points, input_points_neg), axis=0)
    input_labels_all = np.concatenate((input_labels, input_labels_neg), axis=0)

    masks, scores, logits = predictor.predict(
        point_coords=input_points_all,
        point_labels=input_labels_all,
        # box=input_box,
        multimask_output=True,
    )

    for score, mask_ in zip(scores, masks):
        if score > sam_score_thres:
            sam_masks_list.append(mask_)

    return sam_masks_list


def calc_masks_with_pos_neg_points_from_objects(sam_masks_list, sam_score_list, box_dict_2d, label, sam_score_thres,
                                                predictor):
    all_points_list = []
    all_labels_list = []
    for id, box2d in box_dict_2d.items():
        if id == int(label):
            all_points_list.append(box2d['point_center_from_mask3d'])
            all_labels_list.append(1)
        else:
            all_points_list.append(box2d['point_center_from_mask3d'])
            all_labels_list.append(0)

    input_points_all_ary = np.asarray(all_points_list)
    input_points_all_ary = np.squeeze(input_points_all_ary, axis=1)
    input_lables_all_ary = np.asarray(all_labels_list)

    masks, scores, logits = predictor.predict(
        point_coords=input_points_all_ary,
        point_labels=input_lables_all_ary,
        # box=input_box,
        multimask_output=True,
    )

    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(image)
    #     show_mask(mask, plt.gca())
    #     show_points(input_points_all_ary, input_lables_all_ary, plt.gca())
    #     plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.show()

    for score, mask_ in zip(scores, masks):
        # if score > sam_score_thres:
        sam_masks_list.append(mask_)
        sam_score_list.append(score)

    return sam_masks_list, sam_score_list


def calc_masks_with_mult_pos_points(sam_masks_list, sam_score_list, box_dict_2d, label, sam_score_thres, predictor,
                                    mask, image,
                                    num_pos_points=2):
    all_points_list = []
    all_labels_list = []
    mask_tmp = np.copy(mask)
    for id, box2d in box_dict_2d.items():
        if id == int(label):
            point = box2d['point_center_from_mask3d']
            all_points_list.append(point)
            all_labels_list.append(1)
            point_ = np.copy(point)
            for i in range(num_pos_points):
                mask_tmp[int(point_[0, 1]), int(point_[0, 0])] = 0

                mask_tmp = np.pad(mask_tmp, 1, mode='constant')
                _, distance = medial_axis(mask_tmp, return_distance=True)all_inst_seg_2d
                distance = distance[1:-1, 1:-1]

                max_point = np.where(distance == np.max(distance))
                input_points_tmp = np.zeros([1, 2])
                input_points_tmp[:, 0] = max_point[1][0]
                input_points_tmp[:, 1] = max_point[0][0]

                all_points_list.append(input_points_tmp)
                all_labels_list.append(1)

                point_ = input_points_tmp

            # Do n times: clear 3x3 area around pos point in mask, calc new distance and take point with new max distance
            # as additional pos point
            # Visualize points!!!

        else:
            all_points_list.append(box2d['point_center_from_mask3d'])
            all_labels_list.append(0)

    input_points_all_ary = np.asarray(all_points_list)
    input_points_all_ary = np.squeeze(input_points_all_ary, axis=1)
    input_lables_all_ary = np.asarray(all_labels_list)

    masks, scores, logits = predictor.predict(
        point_coords=input_points_all_ary,
        point_labels=input_lables_all_ary,
        # box=input_box,
        multimask_output=True,
    )

    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     plt.figure(figsize=(10, 10))
    #     plt.imshow(image)
    #     show_mask(mask, plt.gca())
    #     show_points(input_points_all_ary, input_lables_all_ary, plt.gca())
    #     plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     plt.show()

    for score, mask_ in zip(scores, masks):
        # if score > sam_score_thres:
        sam_masks_list.append(mask_)
        sam_score_list.append(score)

    return sam_masks_list, sam_score_list


def calc_iou(mask1, mask2):
    intersection = np.sum(mask1 * mask2)
    if intersection == 0:
        return 0.0
    union = np.sum(np.logical_or(mask1, mask2))
    return intersection / union


def generate_box_coords_3d(box_item):
    x_shift = .5
    y_shift = .5
    z_shift = .5

    box = np.array([[x_shift, y_shift, z_shift], [-x_shift, y_shift, z_shift], [-x_shift, -y_shift, z_shift],
                    [x_shift, -y_shift, z_shift], [x_shift, y_shift, -z_shift], [-x_shift, y_shift, -z_shift],
                    [-x_shift, -y_shift, -z_shift], [x_shift, -y_shift, -z_shift]])

    box_tensor = torch.Tensor(box).to(device)

    cad_transform_base = box_item.transform3d.to(device)

    box_transformed = cad_transform_base.transform_points(box_tensor)
    box_transformed = box_transformed.cpu().detach().numpy()
    return box_transformed


def main(args):
    config = load_config(args.config)['general']
    SCANNOTATE_PATH = config['SCANNOTATE_PATH']
    SCANNET_base_path = config['SCANNET_base_path']

    scene_list = os.listdir(SCANNET_base_path)
    scene_list.sort()

    # Process only pgn g.t. scenes
    if config['annotate_pgn_only']:
        gt_scenes = get_pgn_annotated_gt_scenes()
        scene_list = [scene for scene in scene_list if scene not in gt_scenes]

    img_scale = 1.
    inst_seg_2d_labels_list = config.getstruct('inst_seg_2d_labels_list')

    sam_checkpoint = config['sam_checkpoint_path']
    model_type = config['sam_model_type']

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    sam_score_thres = 0.9

    for scene_cnt, scene_name in enumerate(scene_list):

        scene_name = scene_name.rstrip()
        print(scene_name)

        pkl_out_path = os.path.join(SCANNOTATE_PATH, config['out_folder'], scene_name)

        if not os.path.exists(os.path.join(pkl_out_path, scene_name + '.pkl')):
            continue
        if os.path.exists(os.path.join(pkl_out_path, 'sam_results_path')):
            continue

        pkl_file = open(os.path.join(pkl_out_path, scene_name + '.pkl'), 'rb')
        scene_obj = pickle.load(pkl_file)

        inst_seg_2d_folder = os.path.join(pkl_out_path, 'all_inst_seg_2d')
        if not os.path.exists(inst_seg_2d_folder):
            assert False

        sam_results_path = os.path.join(pkl_out_path, 'sam_results_path')
        # skel_results_path = os.path.join(pkl_out_path, 'skeleton_vis')
        valid_maps_path = os.path.join(pkl_out_path, 'valid_maps')

        if not os.path.exists(sam_results_path):
            os.makedirs(sam_results_path)

        # if not os.path.exists(skel_results_path):
        #     os.makedirs(skel_results_path)

        if not os.path.exists(valid_maps_path):
            os.makedirs(valid_maps_path)

        box_dict_3d = {}
        for count, box_item in enumerate(scene_obj.obj_annotation_list):
            box_3d = generate_box_coords_3d(box_item)
            box_dict_3d[int(box_item.object_id)] = box_3d

        for count, box_item in enumerate(scene_obj.obj_annotation_list):


            label = int(box_item.object_id)

            # TODO only generate masks for specific classes?
            sem_cls_name = box_item.category_label
            if sem_cls_name not in inst_seg_2d_labels_list:
                continue
            print(box_item.object_id, box_item.category_label)

            view_parameters = box_item.view_params
            inst_seg_2d_from_render = []

            # print(view_parameters)
            # assert False
            for frame_id in view_parameters['frame_ids']:
                if frame_id == '':
                    frame_id = '0'
                mask_tmp_path = os.path.join(inst_seg_2d_folder, str(label) + '_' + str(frame_id) + '.png')
                mask_tmp = cv2.imread(mask_tmp_path)
                mask_tmp = mask_tmp[:, :, 0]
                inst_seg_2d_from_render.append(mask_tmp)

            for frame_cnt, (inst_seg, R, T, frame_id) in enumerate(
                    zip(inst_seg_2d_from_render, view_parameters['R'], view_parameters['T'],
                        view_parameters['frame_ids'])):

                # if frame_id_ == '':
                #     frame_id_ = '0'

                # frame_id = frame_id_.zfill(6)

                depth_path = os.path.join(SCANNET_base_path, scene_name, 'depth',
                                          str(frame_id) + '.png')
                rgb_path = os.path.join(SCANNET_base_path, scene_name, 'color',
                                        str(frame_id) + '.jpg')

                sam_masks_list = []
                sam_score_list = []

                intri = view_parameters['intrinsics']
                img_rgb = cv2.imread(rgb_path)
                img_height = img_rgb.shape[0]
                img_width = img_rgb.shape[1]
                target_height = 480.
                target_width = 640.
                img_rgb = torch.from_numpy(img_rgb / 255.).to(device)
                img_rgb = torch.unsqueeze(img_rgb, 0)
                img_rgb = img_rgb.permute((0, 3, 1, 2))

                img_rgb = F.interpolate(img_rgb,
                                        scale_factor=(img_scale * (target_height / img_height), img_scale *
                                                      (target_width / img_width))
                                        )

                img_rgb_np = img_rgb.cpu().detach().numpy()
                rgb_im_np = np.squeeze(img_rgb_np, 0)
                rgb_im_np = np.moveaxis(rgb_im_np, [0], [2])
                rgb_img_uint8 = (np.copy(rgb_im_np) * 255).astype('uint8')
                rgb_im_np_tmp = np.copy(rgb_im_np)
                rgb_im_np_tmp2 = np.copy(rgb_im_np)

                image = cv2.cvtColor(rgb_img_uint8, cv2.COLOR_BGR2RGB)
                predictor.set_image(image)

                mask = np.zeros_like(inst_seg)
                mask[inst_seg == label] = 1.

                mask_other_instances = np.copy(inst_seg)
                mask_other_instances[inst_seg == box_item.object_id] = 0

                valid_z, valid_xy, valid_xyz, depth_unknown_mask, min_depth, valid_maps_stacked, valid_3dbox = calc_valid_masks_for_2d(
                    inst_seg, R, T,
                    depth_path, label, img_scale, box_dict_3d[box_item.object_id],
                    intri, rgb_im_np_tmp2, inst_seg_2d_folder,
                    frame_cnt, label, target_height, target_width)

                # Visualization of maps
                cv2.imwrite(os.path.join(valid_maps_path, str(label) + '_' + str(frame_id) + '.jpg'),
                            valid_maps_stacked)

                # For each frame with R,T, transform all 3d boxes into image space, and extract 2d boxes,
                # similar as in calc_valid_masks_for_2d()

                box_dict_2d, all_box2d_list_from_3dbox, all_box2d_list_from_3dmask = calc_all_boxes_2d(inst_seg, R, T,
                                                                                                       depth_path,
                                                                                                       label, img_scale,
                                                                                                       intri,
                                                                                                       rgb_im_np_tmp2,
                                                                                                       inst_seg_2d_folder,
                                                                                                       frame_cnt, label,
                                                                                                       target_height,
                                                                                                       target_width,
                                                                                                       frame_id,
                                                                                                       box_dict_3d)

                # SAM with multiple objects - target box from reprojected 3d box, boxes of other objects from
                # reprojected 3d mask
                sam_masks_list, sam_score_list = calc_masks_with_multiple_boxes(sam_masks_list, sam_score_list,
                                                                                box_dict_2d, predictor,
                                                                                image, sam_score_thres, label)

                # SAM with target point in center of mask2d_from_mask3d
                sam_masks_list, sam_score_list = calc_masks_with_center_point(sam_masks_list, sam_score_list,
                                                                              predictor, sam_score_thres, box_dict_2d,
                                                                              label, mask, num_pos_points=2)

                ## SAM with target point in center of mask2d_from_mask3d + neg points sampled around target object
                # sam_masks_list = calc_masks_with_pos_neg_points(sam_masks_list, predictor, valid_xyz,
                #                                                 depth_unknown_mask, label,
                #                    sam_score_thres, mask, box_dict_2d)

                # SAM with target point in center of mask2d_from_mask3d + neg points from center from
                # mask2d_from_mask3d of other objects
                sam_masks_list, sam_score_list = calc_masks_with_pos_neg_points_from_objects(sam_masks_list,
                                                                                             sam_score_list,
                                                                                             box_dict_2d, label,
                                                                                             sam_score_thres, predictor)

                # SAM with multiple target points + neg points from center from
                # mask2d_from_mask3d of other objects
                sam_masks_list, sam_score_list = calc_masks_with_mult_pos_points(sam_masks_list, sam_score_list,
                                                                                 box_dict_2d, label,
                                                                                 sam_score_thres, predictor, mask,
                                                                                 image)

                sam_masks_list_new = []

                # Remove masks with low overlap to reference mask
                for mask_cnt, (sam_mask, sam_score) in enumerate(zip(sam_masks_list, sam_score_list)):

                    if sam_score < sam_score_thres:
                        continue

                    # TODO add geometric refinement for SAM masks
                    sam_mask[np.logical_not(np.logical_or(valid_xyz, depth_unknown_mask))] = 0

                    iou_score = calc_iou(sam_mask, mask)

                    if iou_score > .4:
                        sam_masks_list_new.append(sam_mask)

                # add rendered reference mask
                sam_masks_list_new.append(mask)

                # # Visualize all valid sam_masks
                prediction_stacked = vis_sam_masks(sam_masks_list_new, rgb_im_np_tmp)

                cv2.imwrite(os.path.join(sam_results_path, str(label) + '_' + str(frame_id) + '_vis_all.jpg'),
                            prediction_stacked)

                sam_mask_final = np.zeros_like(mask)

                # Voting scheme -  fuse all valid SAM masks to one final mask using pixel-wise voting
                if len(sam_masks_list_new) == 1:
                    sam_mask_final = sam_masks_list_new[0]
                else:
                    mask_thres = int((len(sam_masks_list_new) / 2))
                    masks_vote = np.sum(np.asarray(sam_masks_list_new), axis=0)
                    sam_mask_final[masks_vote >= mask_thres] = 1

                # TODO use geometric mask refinement
                #  Remove mask pixel which are not part of the reprojected 3D bounding box
                #sam_mask_final[np.logical_not(valid_3dbox)] = 0
                # TODO or use another alternative which :
                #  Remove mask pixel which are not part of the reprojected 3D bounding box (valid_xyz)
                #  or not part of missing depth value map (depth_unknown_mask)
                #sam_mask_final[np.logical_not(np.logical_or(valid_xyz, depth_unknown_mask))] = 0

                rgb_im_copy = np.copy(rgb_im_np_tmp)
                inst_seg_2d_predicted = np.copy(sam_mask_final)
                v_depth_img = np.zeros_like(rgb_im_copy)
                v_depth_img[inst_seg_2d_predicted == 1.] = (0, 255, 0)
                img_mask_pred = ((rgb_im_copy * 255.) * 0.5 + (v_depth_img) * 0.5)

                cv2.imwrite(os.path.join(sam_results_path, str(label) + '_' + str(frame_id) + '_vis_final.jpg'),
                            img_mask_pred)

                cv2.imwrite(os.path.join(sam_results_path, str(label) + '_' + str(frame_id) + '.png'),
                            sam_mask_final * 255)

if __name__ == '__main__':
    main(parser.parse_args())

import os

def get_pgn_annotated_gt_scenes():
    scenes = []
    for category in ['chair','sofa','table']:
        gt_scenes_path = '../sp_gt_annotations/' + category

        category_scenes = os.listdir(gt_scenes_path)
        category_scenes.sort()

        scenes.extend(category_scenes)

    return scenes
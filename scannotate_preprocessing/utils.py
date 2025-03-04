import zipfile
import os
import json
import numpy as np
import math as m
import open3d as o3d



COLOR_DETECTRON2 = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        # 0.300, 0.300, 0.300,
        0.000, 0.667, 0.000, #0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.000, 0.000, 1.000,
        0.000, 1.000, 0.000,
        0.749, 0.749, 0.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        51/255., 187/255., 255/255.,#1.000, 0.667, 0.000, ##51, 187, 255
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        # 0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        # 0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        #0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        # 0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        # 0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.857, 0.857, 0.857,
        # 1.000, 1.000, 1.000
    ]).astype(np.float32).reshape(-1, 3)# * 255

SEMANTIC_IDXS = np.array([0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14, 15, 16,17,18,19,20,21,22,23, 24,25,26,27, 28,29,
                          30, 31, 32, 33, 34,35, 36,37,38, 39,40])

SEMANTIC_NAMES = np.array(['unannotated', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                           'bookshelf', 'picture', 'counter', 'blinds','desk', 'shelves', 'curtain', 'dresser','pillow',
                           'mirror','floor_mat','clothes','ceiling','book','refridgerator','television','paper','towel',
                           'shower curtain','box','whiteboard','person','night stand',
                           'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture','otherprop'])
CLASS_COLOR = {
    'unannotated': [0, 0, 0],
    'wall': [143, 223, 142],
    'floor': [171, 198, 230],
    'cabinet': [0, 120, 177],
    'bed': [255, 188, 126],
    'chair': [189, 189, 57],
    'sofa': [144, 86, 76],
    'table': [255, 152, 153],
    'door': [222, 40, 47],
    'window': [197, 176, 212],
    'bookshelf': [150, 103, 185],
    'picture': [200, 156, 149],
    'counter': [0, 190, 206],
    'desk': [252, 183, 210],
    'curtain': [219, 219, 146],
    'refridgerator': [255, 127, 43],
    'bathtub': [234, 119, 192],
    'shower curtain': [150, 218, 228],
    'toilet': [0, 160, 55],
    'sink': [110, 128, 143],
    'otherfurniture': [80, 83, 160],
    'blinds':[ 211, 47, 47],
    'shelves':[ 0, 255, 0],
    'dresser':[123, 31, 162],
    'pillow':[81, 45, 168],
    'mirror':[ 48, 63, 159 ],
    'floor_mat':[25, 118, 210],
    'clothes':[2, 136, 209],
    'ceiling':[ 153, 51, 102],
    'book':[0, 121, 107],
    'television':[56, 142, 60 ],
    'paper':[104, 159, 56],
    'towel':[175, 180, 43 ],
    'box':[251, 192, 45],
    'whiteboard':[255, 160, 0],
    'person':[245, 124, 0],
    'night stand':[230, 74, 25],
    'lamp':[93, 64, 55],
    'bag':[97, 97, 97],
    'otherstructure':[ 84, 110, 122 ],
    'otherprop':[255, 255, 102]
}

furniture_list = ['display','trash bin', 'bench','piano' ,'washer']
prop_list = ['printer','basket','flowerpot','laptop','clock','faucet','bowl',
             'keyboard','microwaves','guitar']
structure_list = ['stove','dishwasher']
cabinet_list = ['cabinet','file cabinet']

SEMANTIC_IDX2NAME = {0: 'unannotated',1: 'wall', 2: 'floor', 3: 'cabinet', 4: 'bed', 5: 'chair', 6: 'sofa', 7:
                'table', 8: 'door', 9: 'window', 10: 'bookshelf', 11: 'picture',
                12: 'counter', 13: 'blinds', 14: 'desk', 15:'shelves', 16: 'curtain', 17:'dresser', 18:'pillow',
                     19:'mirror', 20:'floor_mat', 21:'clothes', 22:'ceiling', 23:'book',24: 'refridgerator',
                     25:'television', 26:'paper', 27:'towel', 28: 'shower curtain', 29:'box', 30:'whiteboard',
                     31:'person', 32:'night stand', 33: 'toilet',  34: 'sink', 35:'lamp', 36: 'bathtub', 37:'bag',
                     38:'otherstructure', 39: 'otherfurniture', 40:'otherprop'}
                    #38: structure_list, 39:furniture_list, 40: prop_list}



MSEG_SEMANTIC_IDX2NAME = {0: 'unannotated',1: 'wall', 2: 'floor', 3: 'cabinet', 4: 'bed', 5: 'chair', 6: 'sofa', 7:
                'table', 8: 'door', 9: 'window', 10: 'bookshelf', 11: 'picture',
                12: 'counter', 13: 'blinds', 14: 'desk', 15:'shelves', 16: 'curtain', 17:'dresser', 18:'pillow',
                     19:'mirror', 20:'floor_mat', 21:'clothes', 22:'ceiling', 23:'book',24: 'refridgerator',
                     25:'television', 26:'paper', 27:'towel', 28: 'shower curtain', 29:'box', 30:'whiteboard',
                     31:'person', 32:'night stand', 33: 'toilet',  34: 'sink', 35:'lamp', 36: 'bathtub', 37:'bag',
                    38:'otherstructure', 39: 'otherfurniture', 40:'otherprop', 41:'trash bin', 42:'washer', 43:'display',
                    44:'microwaves', 45:'stove', 46:'dishwasher', 47:'laptop', 48:'bench', 49:'piano', 50:'printer',
                    51:'basket', 52:'flowerpot', 53:'clock', 54:'faucet', 55:'bowl', 56:'keyboard', 57:'guitar',
                          58:'motorbike', 59:'fireplace', 60:'file cabinet' }

shapenet_category_dict = {'airplane': '02691156', 'trash bin': '02747177', 'bag': '02773838', 'basket': '02801938',
                          'bathtub': '02808440', 'bed': '02818832', 'bench': '02828884', 'birdhouse': '02843684',
                          'bookshelf': '02871439', 'bottle': '02876657', 'bowl': '02880940', 'bus': '02924116',
                          'cabinet': '02933112', 'camera': '02942699', 'can': '02946921', 'cap': '02954340',
                          'car': '02958343', 'cellphone': '02992529', 'chair': '03001627', 'clock': '03046257',
                          'keyboard': '03085013', 'dishwasher': '03207941', 'display': '03211117',
                          'earphone': '03261776', 'faucet': '03325088', 'file cabinet': '03337140', 'guitar': '03467517',
                          'helmet': '03513137', 'jar': '03593526', 'knife': '03624134', 'lamp': '03636649',
                          'laptop': '03642806', 'loudspeaker': '03691459', 'mailbox': '03710193',
                          'microphone': '03759954', 'microwaves': '03761084', 'motorbike': '03790512',
                          'mug': '03797390', 'piano': '03928116', 'pillow': '03938244', 'pistol': '03948459',
                          'flowerpot': '03991062', 'printer': '04004475', 'remote': '04074963', 'rifle': '04090263',
                          'rocket': '04099429', 'skateboard': '04225987', 'sofa': '04256520', 'stove': '04330267',
                          'table': '04379243', 'telephone': '04401088', 'tower': '04460130', 'train': '04468005',
                          'watercraft': '04530566', 'washer': '04554684','desk':'03179701', 'dresser':'03237340',
                          'pillow':'03938244', 'bed cabinet': '20000008'}

SCANNET_2_SHAPENET = {
    'unannotated': 'unannotated',
    'wall': 'unannotated',
    'floor': 'unannotated',
    'cabinet': 'cabinet',
    'bed': 'bed',
    'chair': 'chair',
    'sofa': 'sofa,couch,lounge',
    'table': 'table',
    'door': 'unannotated',
    'window': 'unannotated',
    'bookshelf': 'bookshelf',
    'picture': 'unannotated', #ShapeNetSem
    'counter': 'counter',
    'desk': 'desk',
    'curtain': 'unannotated', # ShapeNetSem
    'refridgerator': 'Refrigerator', # ShapeNetSem
    'bathtub': 'bathtub,bathing tub,bath,tub',
    'shower curtain': 'unannotated',# ShapeNetSem
    'toilet': 'Toilet', # ShapeNetSem
    'sink': 'Sink', # ShapeNetSem
    'otherfurniture': 'unannotated',
    'blinds': 'unannotated',
    'shelves': 'bookshelf',
    'dresser': 'dresser',
    'pillow': 'pillow',
    'mirror': 'unannotated', #ShapeNetSem
    'floor_mat': 'unannotated',
    'clothes': 'unannotated',
    'ceiling': 'unannotated',
    'book': 'unannotated', #ShapeNetSem
    'television': 'display,video display',
    'paper': 'unannotated',
    'towel': 'unannotated',
    'box': 'unannotated',
    'whiteboard': 'unannotated',
    'person': 'unannotated',
    'night stand': 'bed cabinet',
    'lamp': 'lamp',
    'bag': 'bag',
    'otherstructure': 'unannotated',
    'otherprop': 'unannotated'
}

SYNSET_DICT_DIR = '/home/stefan/PycharmProjects/Scannet_Total3D/DATA/'

chair_dummy_list = ['/media/stefan/SSD_2/ShapeNet/ShapeNet_norm_scaled/chair/bdc892547cceb2ef34dedfee80b7006.obj',
                    '/media/stefan/SSD_2/ShapeNet/ShapeNet_norm_scaled/chair/2bf05f8a84f0a6f33002761e7a3ba3bd.obj',
                    '/media/stefan/SSD_2/ShapeNet/ShapeNet_norm_scaled/chair/460eef2b90867427d9fad8aba2c312b7.obj',
                    '/media/stefan/SSD_2/ShapeNet/ShapeNet_norm_scaled/chair/8d18fba375d0545edbbc9440457e303e.obj',
                    '/media/stefan/SSD_2/ShapeNet/ShapeNet_norm_scaled/chair/5fdb10483f79355581f5ac91b0c9e99b.obj']

chair_height_dummy_list = ['/media/stefan/SSD_2/ShapeNet/ShapeNet_norm_scaled/chair/bdc892547cceb2ef34dedfee80b7006.obj',
                    '/media/stefan/SSD_2/ShapeNet/ShapeNet_norm_scaled/chair/2bf05f8a84f0a6f33002761e7a3ba3bd.obj',
                    '/media/stefan/SSD_2/ShapeNet/ShapeNet_norm_scaled/chair/460eef2b90867427d9fad8aba2c312b7.obj',
                    '/media/stefan/SSD_2/ShapeNet/ShapeNet_norm_scaled/chair/8d18fba375d0545edbbc9440457e303e.obj',
                    '/media/stefan/SSD_2/ShapeNet/ShapeNet_norm_scaled/chair/5fdb10483f79355581f5ac91b0c9e99b.obj',
                    '/media/stefan/SSD_2/ShapeNet/ShapeNet_norm_scaled/chair/9faecbe3bded39c4efed9665e3f75336.obj']

box_dummy_list = ['/media/stefan/SSD_2/ShapeNet/ShapeNet_norm_scaled/refridgerator/4cd795e6f39ba2ff241e46a406e96e16.obj']


def Ry(theta):
    return np.matrix([[m.cos(theta), 0, m.sin(theta)],
                      [0, 1, 0],
                      [-m.sin(theta), 0, m.cos(theta)]])

def Rx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def Rz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

def alignPy3D_format(pclMesh,T_mat):
    verts = np.array(pclMesh.verts_list()[0])
    newVerts = np.ones((verts.shape[0], 4))
    newVerts[:, :3] = verts
    newVerts = newVerts.dot(T_mat.T)
    pclMesh.vertices = o3d.utility.Vector3dVector(newVerts[:, :3])
    ar = np.asarray(pclMesh.vertices)
    return pclMesh

def transform_ScanNet_to_py3D():
    rot_tmp1 = Rx(np.deg2rad(-90))
    rot_tmp2 = Ry(np.deg2rad(-90))
    rot3 = np.asarray(np.dot(rot_tmp2, rot_tmp1))
    T = np.eye(4)
    T[:3, :3] = rot3
    return T

def transform_ARKIT_to_py3D():
    rot_tmp1 = Rx(np.deg2rad(-90))
    rot_tmp2 = Ry(np.deg2rad(-180))
    rot3 = np.asarray(np.dot(rot_tmp2, rot_tmp1))
    T = np.eye(4)
    T[:3, :3] = rot3
    return T

def alignPclMesh(pclMesh,axis_align_matrix=np.eye(4),T=np.eye(4)):

    if isinstance(pclMesh, o3d.geometry.TriangleMesh):
        verts = np.array(pclMesh.vertices)
        newVerts = np.ones((verts.shape[0],4))
        newVerts[:,:3] = verts
        newVerts = newVerts.dot(axis_align_matrix.T)
        newVerts = newVerts.dot(T.T)
        pclMesh.vertices = o3d.utility.Vector3dVector(newVerts[:,:3])
    elif isinstance(pclMesh, o3d.geometry.PointCloud):
        points = np.array(pclMesh.points)
        newPoints = np.ones((points.shape[0], 4))
        newPoints[:, :3] = points
        newPoints = newPoints.dot(axis_align_matrix.T)
        newPoints = newPoints.dot(T.T)
        pclMesh.points = o3d.utility.Vector3dVector(newPoints[:, :3])
    elif isinstance(pclMesh,np.ndarray):
        points = pclMesh
        newPoints = np.ones((points.shape[0], 4))
        newPoints[:, :3] = points
        newPoints = newPoints.dot(axis_align_matrix.T)
        newPoints = newPoints.dot(T.T)
        return newPoints
        #pclMesh.points = o3d.utility.Vector3dVector(newPoints[:, :3])
    else:
        raise NotImplementedError

    return pclMesh

def yaw_pitch_roll_from_R(cam_R):
    '''
    get the yaw, pitch, roll angle from the camera rotation matrix.
    :param cam_R: Camera orientation. R:=[v1, v2, v3], the three column vectors respectively denote the toward, up,
    right vector relative to the world system.
    Hence, the R = R_y(yaw)*R_z(pitch)*R_x(roll).
    :return: yaw, pitch, roll angles.
    '''
    yaw = np.arctan(-cam_R[2][0] / cam_R[0][0])
    pitch = np.arctan(cam_R[1][0] / np.sqrt(cam_R[0][0] ** 2 + cam_R[2][0] ** 2))
    roll = np.arctan(-cam_R[1][2] / cam_R[1][1])

    return yaw, pitch, roll

def R_from_yaw_pitch_roll(yaw, pitch, roll):
    '''
    Retrieve the camera rotation from yaw, pitch, roll angles.
    Camera orientation. R:=[v1, v2, v3], the three column vectors respectively denote the toward, up,
    right vector relative to the world system.

    Hence, the R = R_y(yaw)*R_z(pitch)*R_x(roll).
    '''
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(yaw) * np.cos(pitch)
    R[0, 1] = np.sin(yaw) * np.sin(roll) - np.cos(yaw) * np.cos(roll) * np.sin(pitch)
    R[0, 2] = np.cos(roll) * np.sin(yaw) + np.cos(yaw) * np.sin(pitch) * np.sin(roll)
    R[1, 0] = np.sin(pitch)
    R[1, 1] = np.cos(pitch) * np.cos(roll)
    R[1, 2] = - np.cos(pitch) * np.sin(roll)
    R[2, 0] = - np.cos(pitch) * np.sin(yaw)
    R[2, 1] = np.cos(yaw) * np.sin(roll) + np.cos(roll) * np.sin(yaw) * np.sin(pitch)
    R[2, 2] = np.cos(yaw) * np.cos(roll) - np.sin(yaw) * np.sin(pitch) * np.sin(roll)
    return R

def downloadScanNetScene(sceneID, scanDir='outputs'):
    filesToDload = ['_vh_clean_2.ply', '.txt', '_vh_clean_2.labels.ply']

    print('Downloading ScanNet V2 Scan %s...'%sceneID)
    scenesList = [sceneID]

    for scene in scenesList:
        for file in filesToDload:
            if not os.path.exists(os.path.join(scanDir, 'scans', scene, scene+file)):
                os.system('python2 download-scannet.py -o %s --id %s --type %s'%(scanDir, scene, file))
            if 'zip' in file:
                if (not os.path.exists(os.path.join(scanDir, 'scans', scene, scene+file[:-4]))) \
                    and (os.path.exists(os.path.join(scanDir, 'scans', scene, scene+file))):
                    with zipfile.ZipFile(os.path.join(scanDir, 'scans', scene, scene+file), 'r') as zip_ref:
                        zip_ref.extractall(os.path.join(scanDir, 'scans', scene, scene+file[:-4]))
                        print('Unizipping %s'%(scene))


def downloadValScenes(scanDir='outputs'):
    with open('scannetv2_val.txt','r') as f:
        sceneIDs = f.readlines()
    sceneIDs = [s.strip() for s in sceneIDs if '_00' in s]
    for i,scene in enumerate(sceneIDs):
        print('Downloading Scene %d of %d'%(i, len(sceneIDs)))
        if os.path.exists(os.path.join(scanDir, 'scans', scene, 'monte_carlo', 'FinalCandidates.pickle')):
            downloadScanNetScene(scene, scanDir)

def jsonWrite(filename, data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)


def jsonRead(filename):
    with open(filename, 'r') as infile:
        return json.load(infile)



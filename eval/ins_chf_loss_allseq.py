# calculate chamfer distance around instace
import os
from os.path import join, basename
from glob import glob
import argparse
from tqdm import tqdm
import yaml
import numpy as np
import torch

from typing import Tuple
from numba import njit
from chamferdist import ChamferDistance

# load Semantic KITTI class info
with open("semantic-kitti.yaml", 'r') as stream:
    semkittiyaml = yaml.safe_load(stream)
SemKITTI_label_name = dict()
for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
    SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]
things = ['car', 'truck', 'bicycle', 'motorcycle', 'bus', 'person', 'bicyclist', 'motorcyclist']
stuff = ['road', 'sidewalk', 'parking', 'other-ground', 'building', 'vegetation', 'trunk', 'terrain', 'fence', 'pole', 'traffic-sign']
things_ids = [] # original semantic labels that is thing class
for i in sorted(list(semkittiyaml['labels'].keys())):
    if SemKITTI_label_name[semkittiyaml['learning_map'][i]] in things:
        things_ids.append(i)
print(things_ids) # [10, 11, 13, 15, 16, 18, 20, 30, 31, 32, 252, 253, 254, 255, 256, 257, 258, 259]

# class mapping
with open("semantic-kitti.yaml", 'r') as stream:
    doc = yaml.safe_load(stream)
    all_labels = doc['labels']
    learning_map_inv_ = doc['learning_map_inv']
    learning_map_ = doc['learning_map']
    learning_map = np.zeros((np.max([k for k in learning_map_.keys()]) + 1), dtype=np.int32)
    for k, v in learning_map_.items():
        learning_map[k] = v

    learning_map_inv = np.zeros((np.max([k for k in learning_map_inv_.keys()]) + 1), dtype=np.int32)
    for k, v in learning_map_inv_.items():
        learning_map_inv[k] = v

##################################################################################################################################
# chamfer distance utility
##################################################################################################################################

# https://github.com/danielhavir/voxelize3d
@njit
def voxelize_jit(
        points: np.ndarray,
        voxel_size: np.ndarray,
        grid_range: np.ndarray,
        #max_points_in_voxel: int = 60,
        #max_num_voxels: int = 20000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Numba-friendly version of voxelize
    :param points: (num_points, num_features), first 3 elements must be <x>, <y>, <z>
    :param voxel_size: (3,) - <width>, <length>, <height>
    :param grid_range: (6,) - <min_x>, <min_y>, <min_z>, <max_x>, <max_y>, <max_z>
    :param max_points_in_voxel:
    :param max_num_voxels:
    :return: tuple (
        voxels (num_voxels, max_points_in_voxels, num_features),
        coordinates (num_voxels, 3),
        num_points_per_voxel (num_voxels,)
    )
    '''
    points_copy = points.copy()
    grid_size = np.floor((grid_range[3:] - grid_range[:3]) / voxel_size).astype(np.int32)

    # assign voxel id to each point
    coor = np.floor((points_copy[:, :3] - grid_range[:3]) / voxel_size).astype(np.int32)
    # filter points that is outside the required grid range
    mask = np.logical_and(np.logical_and((coor[:, 0] >= 0) & (coor[:, 0] < grid_size[0]),
                                         (coor[:, 1] >= 0) & (coor[:, 1] < grid_size[1])),
                          (coor[:, 2] >= 0) & (coor[:, 2] < grid_size[2]))
    #points_copy = coor[mask, :]
    points_copy = points_copy[mask, :]

    return points_copy

def chf_loss(upsampled_pts, gt_pts):
    '''
    upsampled_pts shape: [N, 3], np array
    gt_pts shape: [N, 3], np array
    '''
    chamferDist = ChamferDistance()
    upsampled_pts = torch.from_numpy(upsampled_pts).type(torch.DoubleTensor).cuda().unsqueeze(0)
    gt_pts = torch.from_numpy(gt_pts).type(torch.DoubleTensor).cuda().unsqueeze(0)
    #print("gt_pts.shape: ", gt_pts.shape)
    #print("upsampled_pts.shape: ", upsampled_pts.shape)

    forward = chamferDist(upsampled_pts, gt_pts, reduction=None)
    backward = chamferDist(gt_pts, upsampled_pts, reduction=None)
    #forward, backward, idx1, idx2 = chamferDist(upsampled_pts, gt_pts)
    #print('forward: ', forward)
    #print('backward: ', backward)
    chamfer_loss = torch.mean(forward) + torch.mean(backward)
    #chamfer_loss = torch.mean(forward)
    #chamfer_loss = torch.mean(backward)

    return chamfer_loss


##################################################################################################################################
# common point operations
##################################################################################################################################

def find_max_min(points: np.ndarray,
                 find_max: bool = True,
                 find_min: bool = True):
    '''points: [N, 3]'''
    if find_max:
        print(f'max_x: {np.max(points, axis=0)[0]}, max_y: {np.max(points, axis=0)[1]}, max_z: {np.max(points, axis=0)[2]}')
    if find_min:
        print(f'min_x: {np.min(points, axis=0)[0]}, min_y: {np.min(points, axis=0)[1]}, min_z: {np.min(points, axis=0)[2]}')

def filter_radius(points: np.ndarray,
                  radius: int,
                  center: np.ndarray):
    '''
    ## param
    - points: [N, 3]
    - center: [N, 3]
    ## return
    - mask: [N, ]
    '''
    mask = np.where(np.sum((points - center)**2, axis=1) < (radius**2))[0]
    return mask


##################################################################################################################################
# main process
##################################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ins_path', type=str, default='/media/1TB_SSD/all_seq_result/', help='predict scene path')
    parser.add_argument('--gt_path', type=str, default='/media/1TB_SSD/Sem_Kitti/dataset/sequences/', help='ground truth path')
    args = parser.parse_args()

    ins_path = args.ins_path
    gt_path = args.gt_path

    seqs = [str(i).zfill(2) for i in range(1, 11)] # 01 ~ 10
    #models = ['Ins_aux_result', 'npt_result', 'pugcn', 'mpu']
    models = ['Ins_aux_result']

    # filter condition
    radius = 50 # meter
    ins_out_range = 0.3
    voxel_grid_size = 0.1 # no use here
    voxel_size = np.array([voxel_grid_size, voxel_grid_size, voxel_grid_size]) # no use here
    # 1: car, 2: bicycle, 3: motorcycle
    # 4: truck, 5: other-vehicle
    # 6: person, 7: bicyclist, 8: motorcyclist
    Ins_sem = [i for i in range(1, 9)] # 1~8
    # road: 9 ~ 12
    road_cls = [9, 10, 11, 12]

    for seq in seqs:
        for model in models:
            file_path = join(ins_path, seq, model)
            # find file list
            file_list = sorted(glob(join(file_path, '*.xyz')))

            for class_i in Ins_sem:
                require_sem = [class_i]        
                ins_chf = 0
                cnt = 0
                for ins_i in tqdm(file_list, total=len(file_list)):
                    # read points
                    ins = np.loadtxt(ins_i)
                    if model == 'npt_result':
                        gt_file = join(gt_path, seq, 'velodyne', basename(ins_i)[:-9]+'.bin')
                    else:
                        gt_file = join(gt_path, seq, 'velodyne', basename(ins_i)[:-4]+'.bin')
                    gt = np.fromfile(gt_file, dtype=np.float32).reshape((-1, 4))
                    gt = gt[:, :3]
                    # read lbls
                    if model == 'npt_result':
                        lbl_path_i = join(gt_path, seq, 'labels', basename(ins_i)[:-9]+'.label')
                    else:
                        lbl_path_i = join(gt_path, seq, 'labels', basename(ins_i)[:-4]+'.label')
                    lbl = np.fromfile(lbl_path_i, dtype=np.int32).reshape((-1, 1))
                    sem_lbl = lbl & 0xFFFF  # semantic label in lower half
                    sem_lbl = learning_map[sem_lbl]

                    # filter point within radius
                    center = np.zeros((1, 3), dtype=np.float)
                    ins_mask = filter_radius(ins, radius, center)
                    gt_mask = filter_radius(gt, radius, center)
                    ins_points = ins[ins_mask, :3]
                    gt_points = gt[gt_mask, :3]
                    #print(ins_points.shape)
                    #print(gt_points.shape)

                    # remove road plane (average height)
                    all_road_z = np.zeros((0))
                    for sem_cls in road_cls:
                        sem_mask_gt = np.where(sem_lbl == sem_cls)[0]
                        road_pts_z = gt[sem_mask_gt, 2]
                        all_road_z = np.concatenate((all_road_z, road_pts_z))
                    mean_road_z = np.mean(all_road_z) + 0.1
                    #print(max_road_z)

                    # find location of instance from gt lbl
                    ins_cls_loc = []
                    # filter sem label
                    for sem_cls in require_sem:
                        sem_mask_gt = np.where(sem_lbl == sem_cls)[0]
                        sem_cls_ins_lbl = lbl[sem_mask_gt]
                        unique_ins_cls = np.unique(sem_cls_ins_lbl)
                        sem_cls_loc = []
                        for ins_cls in unique_ins_cls:
                            ins_cls_mask = np.where(lbl == ins_cls)[0]
                            ins_cls_pts = gt[ins_cls_mask, :3]
                            max_xyz = np.max(ins_cls_pts, axis=0)
                            min_xyz = np.min(ins_cls_pts, axis=0)
                            # fit xyz to the voxel grid size
                            max_xyz = (np.ceil(max_xyz/voxel_grid_size)*voxel_grid_size+ins_out_range).reshape(-1)
                            min_xyz = (np.floor(min_xyz/voxel_grid_size)*voxel_grid_size-ins_out_range).reshape(-1)
                            #min_xyz[2] += 0.5
                            min_xyz[2] = mean_road_z
                            sem_cls_loc.append(np.concatenate((min_xyz, max_xyz)))
                        ins_cls_loc.append(sem_cls_loc)
                    # voxelize
                    for sem_cls_xyz in ins_cls_loc:
                        for sem_xyz in sem_cls_xyz:
                            scene_range = sem_xyz

                            coor_ins = voxelize_jit(ins_points, voxel_size, scene_range)
                            unique_voxel_ins = np.unique(coor_ins, axis=0)
                            #print(len(unique_voxel_ins))
                            coor_gt = voxelize_jit(gt_points, voxel_size, scene_range)
                            unique_voxel_gt = np.unique(coor_gt, axis=0)
                            #print(len(unique_voxel_gt))

                            if len(unique_voxel_ins) == 0 or len(unique_voxel_gt) == 0:
                                continue

                            # calculate miou: intersection / union
                            ins_chf_i = chf_loss(unique_voxel_ins, unique_voxel_gt).item()
                            ins_chf += ins_chf_i
                            #print('ins_miou: ', ins_miou_i)
                            cnt += 1
                            #if len(unique_voxel_gt) > 20 and cnt >= 20:
                            #    np.savetxt('./ins.xyz', coor_ins, fmt='%.6f')
                            #    np.savetxt('./gt.xyz', coor_gt, fmt='%.6f')
                            #raise ValueError

                if cnt == 0:
                    ins_chf = 0
                else:
                    ins_chf = ins_chf / cnt

                print(f'Mean chf of {class_i}: {ins_chf}')
                print(f'Total number of {class_i}: {cnt}')

                txt_save_name = join(ins_path, seq, model+'_'+str(class_i)+'.txt')
                with open(txt_save_name, 'w') as f:
                    f.write(f'Mean ins iou of {class_i}: {ins_chf}\n')
                    f.write(f'Total number of {class_i}: {cnt}')
    

    


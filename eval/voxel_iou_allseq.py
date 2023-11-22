import os
from os.path import join, basename
from glob import glob
import time
import argparse
from tqdm import tqdm
import numpy as np
import torch

from typing import Tuple
from numba import njit

##################################################################################################################################
# voxel miou utility
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
    coor = coor[mask, ::-1]
    '''
    points_copy = points_copy[mask]
    assert points_copy.shape[0] == coor.shape[0]

    coor_to_voxelidx = np.full((grid_size[2], grid_size[1], grid_size[0]), -1, dtype=np.int32)
    voxels = np.zeros((max_num_voxels, max_points_in_voxel, points_copy.shape[-1]), dtype=points_copy.dtype)
    coors = np.zeros((max_num_voxels, 3), dtype=np.int32)
    num_points_per_voxel = np.zeros(shape=(max_num_voxels,), dtype=np.int32)

    voxel_num = 0
    for i, c in enumerate(coor):
        voxel_id = coor_to_voxelidx[c[0], c[1], c[2]]
        if voxel_id == -1:
            voxel_id = voxel_num
            if voxel_num > max_num_voxels:
                continue
            voxel_num += 1
            coor_to_voxelidx[c[0], c[1], c[2]] = voxel_id
            coors[voxel_id] = c
        n_pts = num_points_per_voxel[voxel_id]
        if n_pts < max_points_in_voxel:
            voxels[voxel_id, n_pts] = points_copy[i]
            num_points_per_voxel[voxel_id] += 1
    
    return voxels[:voxel_num], coors[:voxel_num], num_points_per_voxel[:voxel_num]
    '''
    return coor

# https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
def find_intersect(A, B):
    '''
    ## param
    - A: [N, 3]
    - B: [N, 3]
    ## return
    - intersect: [intersect_num, 3]
    '''
    nrows, ncols = A.shape
    dtype={
        'names': ['f{}'.format(i) for i in range(ncols)],
        'formats': ncols * [A.dtype]
        }

    intersect = np.intersect1d(A.view(dtype), B.view(dtype))
    intersect = intersect.view(A.dtype).reshape(-1, ncols)
    return intersect

def fins_union(A, B):
    '''
    ## param
    - A: [N, 3]
    - B: [N, 3]
    ## return
    - union: [union_num, 3]
    '''
    A_and_B = np.concatenate((A, B), axis=0)
    union = np.unique(A_and_B, axis=0)
    return union

def miou(A, B):
    '''
    ## param
    - A: [N, 3]
    - B: [N, 3]
    ## return
    - miou_score: intersection / union
    '''
    union = fins_union(A, B)
    intersect = find_intersect(A, B)
    return len(intersect) / len(union)


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
    
    ins_path = '/media/1TB_SSD/all_seq_result/'
    gt_path = '/media/1TB_SSD/Sem_Kitti/dataset/sequences/'

    seqs = [str(i).zfill(2) for i in range(1, 11)] # 01 ~ 10
    #models = ['Ins_aux_result', 'npt_result', 'pugcn', 'mpu']
    models = ['mpu']
    radius = 50 # meter
    voxel_grid_size = 0.2
    voxel_size = np.array([voxel_grid_size, voxel_grid_size, voxel_grid_size])
    scene_range = np.array([-50, -50, -10, 50, 50, 10])
    

    for seq in seqs:
        for model in models:
            file_path = join(ins_path, seq, model)
            # find file list
            file_list = sorted(glob(join(file_path, '*.xyz')))
        
            ins_miou = 0
            for ins_i in tqdm(file_list, total=len(file_list)):
                ins = np.loadtxt(ins_i)
                if model == 'npt_result':
                    gt_file = join(gt_path, seq, 'velodyne', basename(ins_i)[:-9]+'.bin')
                else:
                    gt_file = join(gt_path, seq, 'velodyne', basename(ins_i)[:-4]+'.bin')
                gt = np.fromfile(gt_file, dtype=np.float32).reshape((-1, 4))
                gt = gt[:, :3]

                # filter point within radius
                center = np.zeros((1, 3), dtype=np.float)
                ins_mask = filter_radius(ins, radius, center)
                gt_mask = filter_radius(gt, radius, center)
                ins_points = ins[ins_mask, :3]
                gt_points = gt[gt_mask, :3]
                #print(ins_points.shape)
                #print(gt_points.shape)
                
                # voxelize
                coor_ins = voxelize_jit(ins_points, voxel_size, scene_range)
                unique_voxel_ins = np.unique(coor_ins, axis=0)
                #print(len(unique_voxel_ins))
                coor_gt = voxelize_jit(gt_points, voxel_size, scene_range)
                unique_voxel_gt = np.unique(coor_gt, axis=0)
                #print(len(unique_voxel_gt))

                # calculate miou: intersection / union
                ins_miou_i = miou(unique_voxel_ins, unique_voxel_gt)
                ins_miou += ins_miou_i

            ins_miou = ins_miou / len(file_list)

            print(f'Mean ins iou: {ins_miou}')
            txt_save_name = join(ins_path, seq, model+'_allscene.txt')
            with open(txt_save_name, 'w') as f:
                f.write(f'Mean ins iou: {ins_miou}')




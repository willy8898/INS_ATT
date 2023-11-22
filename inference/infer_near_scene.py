import os
from os.path import join, basename
from glob import glob
import time

from tqdm import tqdm
import numpy as np
import torch

from chamferdist import ChamferDistance


# TO-DO: loss
def chf_loss(upsampled_pts, gt_pts):
    '''
    upsampled_pts shape: [N, 3]
    gt_pts shape: [N, 3]
    '''
    chamferDist = ChamferDistance()
    upsampled_pts = upsampled_pts.unsqueeze(0)
    gt_pts = gt_pts.unsqueeze(0)
    #print("gt_pts.shape: ", gt_pts.shape)
    #print("upsampled_pts.shape: ", upsampled_pts.shape)

    forward = chamferDist(upsampled_pts, gt_pts, reduction=None)
    backward = chamferDist(gt_pts, upsampled_pts, reduction=None)
    #print('forward: ', forward)
    #print('backward: ', backward)
    chamfer_loss = forward.mean() + backward.mean()

    return chamfer_loss

def normalize_point_cloud(pc):
    """
    normalized to [-1, 1]
    pc: [P, 3]
    """
    centroid = np.mean(pc, axis=0, keepdims=True)
    pc = pc - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(pc ** 2, axis=-1)), axis=0, keepdims=True)
    pc = pc / np.expand_dims(furthest_distance, axis=-1)
    return pc, centroid, furthest_distance


if __name__ == '__main__':
    radius = 50 # meter
    ins_path = '/media/1TB_HDD/kitti_ins_aux/KITTI_ins_subscene_result_0/'
    p_path = '/media/1TB_HDD/kitti_ins_aux/KITTI_subscene_result/'
    gt_path = '/media/1TB_SSD/Sem_Kitti/dataset/sequences/08/velodyne/'

    ins_save = '/media/1TB_HDD/kitti_ins_aux/KITTI_ins_subscene_result_50/'
    p_save = '/media/1TB_HDD/kitti_ins_aux/KITTI_subscene_result_50/'
    gt_save = '/media/1TB_HDD/kitti_ins_aux/KITTI_subscene_gt_50/'

    ins_list = sorted(glob(join(ins_path, '*.xyz')))
    p_list = sorted(glob(join(p_path, '*.xyz')))

    ins_loss = 0
    p_loss = 0

    for ins_i, p_i in tqdm(zip(ins_list, p_list), total=len(ins_list)):
        ins = np.loadtxt(ins_i)
        p = np.loadtxt(p_i)
        #print(ins.shape)
        #print(p.shape)
        gt_file = join(gt_path, basename(ins_i)[:-4]+'.bin')
        gt = np.fromfile(gt_file, dtype=np.float32).reshape((-1, 4))
        gt = gt[:, :3]

        # filter point within radius
        center = np.zeros((1, 3), dtype=np.float)
        ins_mask = np.where(np.sum((ins - center)**2, axis=1) < (radius**2))[0]
        p_mask = np.where(np.sum((p - center)**2, axis=1) < (radius**2))[0]
        gt_mask = np.where(np.sum((gt - center)**2, axis=1) < (radius**2))[0]
        #print(ins_mask.shape)
        #print(p_mask.shape)
        #print(gt_mask.shape)

        ins_points = ins[ins_mask, :3]
        p_points = p[p_mask, :3]
        gt_points = gt[gt_mask, :3]
        #print(ins_points.shape)
        #print(p_points.shape)
        #print(gt_points.shape)

        # save predict point cloud
        np.savetxt(join(ins_save, basename(ins_i)), ins_points, fmt='%.6f')
        np.savetxt(join(p_save, basename(p_i)), p_points, fmt='%.6f')
        np.savetxt(join(gt_save, basename(gt_file)[:-4]+'.xyz'), gt_points, fmt='%.6f')

        #ins_points, _, _ = normalize_point_cloud(ins_points)
        ins_points = torch.from_numpy(ins_points).type(torch.FloatTensor).cuda()
        #p_points, _, _ = normalize_point_cloud(p_points)
        p_points = torch.from_numpy(p_points).type(torch.FloatTensor).cuda()
        #gt_points, _, _ = normalize_point_cloud(gt_points)
        gt_points = torch.from_numpy(gt_points).type(torch.FloatTensor).cuda()

        ins_loss_i = chf_loss(ins_points, gt_points)
        ins_loss += ins_loss_i.item()
        print(f'chamfer distance ins: {ins_loss_i.item()}')

        p_loss_i = chf_loss(p_points, gt_points)
        p_loss += p_loss_i.item()
        print(f'chamfer distance p: {p_loss_i.item()}')

        

    avg_ins_loss = ins_loss / len(ins_list)
    print(f'Avg ins chamfer loss: {avg_ins_loss:10.6f}')

    avg_p_loss = p_loss / len(ins_list)
    print(f'Avg p chamfer loss: {avg_p_loss:10.6f}')

    # save chamfer distance result
    with open('./ins_p_chfloss.txt', 'w') as f:
        f.write(f'average ins chamfer distance: {avg_ins_loss:10.8f}\n')
        f.write(f'average p   chamfer distance: {avg_p_loss:10.8f}\n')
        





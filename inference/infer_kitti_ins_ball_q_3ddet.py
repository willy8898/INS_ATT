import os
from os.path import join, basename
import time
import sys
import argparse
import yaml
from glob import glob
import numpy as np
import torch
from tqdm import tqdm
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../code/')

from sklearn.neighbors import KDTree, NearestNeighbors

######  The network and loss are figured out here  ###### 
from chamferdist import ChamferDistance
###### ------ ######
from network_ins_aux import Net_conpu_v7
from pointnet2 import pointnet2_utils as pn2_utils

# kitti sem label mapping
with open('semantic-kitti.yaml', 'r') as stream:
    doc = yaml.safe_load(stream)
    learning_map = doc['learning_map']
    learning_map_ = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
    for k, v in learning_map.items():
        learning_map_[k] = v

def FPS(pts, sample_num):
    pts = torch.from_numpy(pts).unsqueeze(0).cuda()
    upsampled_p_fps_id = pn2_utils.furthest_point_sample(pts.contiguous(), sample_num)
    upsample_result_fps = pn2_utils.gather_operation(pts.permute(0, 2, 1).contiguous(), upsampled_p_fps_id)
    upsample_result_fps = upsample_result_fps.permute(0,2,1) # 1xNx3
    sample_pts = (upsample_result_fps.squeeze(0)).cpu().detach().numpy().astype(np.float32)
    return sample_pts


def dynamic_sample_points(pts, dist_list, total_sample_pts=500):
    '''
    pts: (N, 3)
    dist_list: eg, [0, 10, 20, 30, 40, 50]
    '''
    center = np.array([0,0,0], dtype=np.float32).reshape(1, 3)
    
    mask_list = []
    mask_num = []
    pts_dist = np.sum(np.square(pts[:, :3] - center), axis=1)
    total_valid_num = 0
    for i in range(len(dist_list)-1):
        valid_radius_min = dist_list[i]
        valid_radius_max = dist_list[i+1]
        valid_mask_ = (pts_dist > valid_radius_min**2) & (pts_dist < valid_radius_max**2)
        valid_mask = np.where(valid_mask_)[0].astype(np.int32)
        mask_list.append(valid_mask)
        mask_num.append(len(valid_mask))
        total_valid_num += len(valid_mask)
    
    sample_pts_in_dist = np.zeros((0, 3), dtype=np.float32)
    for valid_num, valid_mask in zip(mask_num, mask_list):
        sample_ratio = valid_num / total_valid_num
        sample_num = int(total_sample_pts*sample_ratio)
        valid_pts = pts[valid_mask, :3]
        sample_pts = FPS(valid_pts, sample_num)
        sample_pts_in_dist = np.concatenate((sample_pts_in_dist, sample_pts), axis=0)
    #print(sample_pts_in_dist.shape)
    return sample_pts_in_dist


def chf_loss(upsampled_pts, gt_pts):
    '''
    upsampled_pts shape: [1, N, 3]
    gt_pts shape: [N, 3]
    '''
    chamferDist = ChamferDistance()
    gt_pts = gt_pts.unsqueeze(0)
    #print("gt_pts.shape: ", gt_pts.shape)
    #print("upsampled_pts.shape: ", upsampled_pts.shape)

    forward = chamferDist(upsampled_pts, gt_pts, reduction=None)
    backward = chamferDist(gt_pts, upsampled_pts, reduction=None)
    #print('forward: ', forward)
    #print('backward: ', backward)
    chamfer_loss = forward.mean() + backward.mean()

    return chamfer_loss

def extract_knn_patch(queries, lbl, pc, k):
    """
    - queries: centroids, [M, C]
    - lbl: feature or instance labels, [P, C]
    - pc: the whole point scene, [P, C]
    - k: max point within final ball query, int
    """
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc)
    knn_idx = knn_search.kneighbors(queries, return_distance=False)
    k_patches = np.take(pc, knn_idx, axis=0)  # M, K, C
    k_lbls = np.take(lbl, knn_idx, axis=0) # M, K, C
    return k_patches, k_lbls

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

def inference(net, pc, lbl_info, patch_num_point=256, patch_num_ratio=3, batch=32, radius=5.5, seed_num=None, sample_points_list=None):
    """
    pc: [P, 3]
    """
    if sample_points_list is None:
        # FPS to get patch
        if seed_num is None:
            seed1_num = int(pc.shape[0] / patch_num_point * patch_num_ratio)
        else:
            seed1_num = seed_num
        
        points = torch.from_numpy(np.expand_dims(pc, axis=0)).cuda() # 1, P, 3
        lbls = lbl_info # P, 1

        # FPS on points
        upsampled_p_fps_id = pn2_utils.furthest_point_sample(points.contiguous(), seed1_num)
        upsample_result_fps = pn2_utils.gather_operation(points.permute(0, 2, 1).contiguous(), upsampled_p_fps_id)
        upsample_result_fps = (upsample_result_fps.permute(0,2,1).squeeze(0)).cpu().detach().numpy().astype(np.float32) # Nx3 
        print(upsample_result_fps.shape)
    else:
        lbls = lbl_info # P, 1
        upsample_result_fps = sample_points_list

    print("number of patches: %d" % upsample_result_fps.shape[0])
    input_list = []
    up_point_list = []

    patches, instances = extract_knn_patch(upsample_result_fps, lbls, pc, patch_num_point)
    centers = upsample_result_fps
    #print(patches.shape)
    #print(instances.shape)

    net.eval()
    no_infer_num = 0
    with torch.no_grad():
        for i in range(len(patches)):
            patch, ins = patches[i, :, :3], instances[i, :, :1]
            #print(patch.shape)
            #print(ins.shape)
            mask = np.sum(np.square(patch[:, :3] - centers[i:i+1, :3]), axis=1) < radius ** 2
            mask_inds = np.where(mask)[0].astype(np.int32)
            patch = patch[mask_inds, :3]
            ins = ins[mask_inds, :1]
            #print(patch.shape)
            #print(ins.shape)
            #print(f'max_x: {np.max(patch, axis=0)[0]}, max_y: {np.max(patch, axis=0)[1]}, max_z: {np.max(patch, axis=0)[2]}')
            #print(f'min_x: {np.min(patch, axis=0)[0]}, min_y: {np.min(patch, axis=0)[1]}, min_z: {np.min(patch, axis=0)[2]}')

            p = torch.from_numpy(np.expand_dims(patch, axis=0)).cuda() # [bs, 256, 3]
            i = torch.from_numpy(np.expand_dims(ins, axis=0)).type(torch.FloatTensor).cuda() # [bs, 256, 1]
            #print(p.shape)
            #print(i.shape)
            
            if p.shape[1] < 20:
                up_point_list.append(p.squeeze(0).cpu().detach().numpy().astype(np.float32))
                no_infer_num += 1
            else:
                # predict
                upsampled_p = net(p, i, is_train=False)
                #print(upsampled_p.shape)
                upsample_point_np = upsampled_p.squeeze(0).cpu().detach().numpy().astype(np.float32)
            
                #up_point_list.append(upsampled_p.squeeze(0))
                up_point_list.append(upsample_point_np)
            
            #raise ValueError('A very specific bad thing happened.')
    print(' no_infer_num: ',  no_infer_num)
    return input_list, up_point_list

def net_infer(net, gt_path, input_path, xyzresult_folder, lbl_path, ori_xyz_folder, chfloss_txt_path='./chf_d.txt'):
    #cnt = 0
    max_cnt = 7481
    #sample_id_list = [x.strip() for x in open('/media/1TB_HDD/KITTI_3Ddet/ImageSets/val.txt').readlines()]
    #sample_id_list = [x.strip() for x in open('/media/1TB_HDD/KITTI_3Ddet/ImageSets/train.txt').readlines()]
    #print(sample_id_list)

    if gt_path is not None:
        total_chf_loss = 0
        for xyz_file, gt_xyz_file in tqdm(zip(input_path, gt_path), total=max_cnt):
            #cnt += 1
            #if str(cnt-1).zfill(6) not in sample_id_list:
            #    continue

            # read point cloud file
            #frame_points_ = np.fromfile(xyz_file, dtype=np.float32).reshape((-1, 4))
            #frame_points = frame_points_[:, :3] # N, 3
            frame_points = np.loadtxt(xyz_file)
            #frame_points, centroid, furthest_distance = normalize_point_cloud(frame_points)
            print(frame_points.shape)
            #gt_points = np.fromfile(gt_xyz_file, dtype=np.float32).reshape((-1, 4))
            #gt_points = gt_points[:, :3]
            gt_points = frame_points
            #gt_points, centroid_gt, furthest_distance_gt = normalize_point_cloud(gt_points)
            gt_points = torch.from_numpy(gt_points).cuda()
            # read instance info
            sem_labels = np.zeros((frame_points.shape[0], 1))
            #lbl_info = np.fromfile(lbl_file, dtype=np.int32).reshape((-1, 1))
            #sem_labels = lbl_info & 0xFFFF  # semantic label in lower half
            #sem_labels = learning_map_[sem_labels]

            ts = time.time()
            # model inference
            input_list, up_point_list = inference(net, frame_points, sem_labels, patch_num_point=10000, seed_num=500)
            #dist_list = [0, 10, 20, 30, 40, 50]
            #sample_pts_in_dist = dynamic_sample_points(frame_points, dist_list, total_sample_pts=500)
            #input_list, up_point_list = inference(net, frame_points, sem_labels, patch_num_point=10000, seed_num=500, sample_points_list=sample_pts_in_dist)
            te = time.time()
            print(f'Infer time: {(te - ts):.6f}s')
            # FPS to 2* point nums
            upsample_result_np = np.concatenate(up_point_list, axis=0)

            upsample_result_th = torch.unsqueeze(torch.from_numpy(upsample_result_np), dim=0).cuda() # 1xNx3  
            #upsample_result_th = torch.unsqueeze(torch.cat(up_point_list, 0), dim=0) # 1xNx3
            #print(upsample_result_th.shape)
            upsampled_p_fps_id = pn2_utils.furthest_point_sample(upsample_result_th.contiguous(), 2*frame_points.shape[0])
            upsample_result_fps = pn2_utils.gather_operation(upsample_result_th.permute(0, 2, 1).contiguous(), upsampled_p_fps_id)
            upsample_result_fps = upsample_result_fps.permute(0,2,1) # 1xNx3
            print(upsample_result_fps.shape)

            # save predict point cloud
            t0 = time.time()
            out_pcd = (upsample_result_fps.squeeze(0)).cpu().detach().numpy().astype(np.float32)
            t1 = time.time()
            print(f'Convert time: {(t1 - t0):.6f}s')
            #out_pcd = (out_pcd*furthest_distance) + centroid
            np.savetxt(join(xyzresult_folder, basename(xyz_file)[:-4]+'_sr.xyz'), out_pcd, fmt='%.6f')

            '''
            # very very important to use .astype('float32')!!
            # or the result is very likely saved wrongly
            out_file = join(xyzresult_folder.replace('_xyz', ''), basename(xyz_file)[:-4]+'.bin')
            out_pcd.astype('float32').tofile(out_file)

            t2 = time.time()
            chf_dist = chf_loss(upsample_result_fps, gt_points)
            print(f'chamfer distance: {chf_dist}')
            total_chf_loss += chf_dist.item()
            t3 = time.time()
            print(f'Loss time: {(t3 - t2):.6f}s')
            '''
            #cnt += 1
            #if cnt >= max_cnt:
            #    break
        '''
        #avg_chf_loss = total_chf_loss / len(gt_path)
        avg_chf_loss = total_chf_loss / max_cnt
        print(f'Avg chamfer loss: {avg_chf_loss:10.6f}')

        # save chamfer distance result
        with open(chfloss_txt_path, 'w') as f:
            f.write(f'average chamfer distance: {avg_chf_loss:10.6f}')
        '''
    
    else:
        #file_id = -1
        cnt = 0
        max_cnt = len(input_path) # how many files want to sr
        #sample_id_list = [x.strip() for x in open('/media/1TB_HDD/KITTI_3Ddet/ImageSets/val.txt').readlines()]
        for xyz_file in tqdm(input_path, total=len(input_path)):
            #file_id += 1
            #if str(file_id).zfill(6) not in sample_id_list:
            #    continue
            
            # read point cloud file
            #frame_points = np.loadtxt(xyz_file).astype(np.float32) # 2048, 3
            #frame_points_ = np.fromfile(xyz_file, dtype=np.float32).reshape((-1, 4))
            frame_points_ = np.loadtxt(xyz_file).astype(np.float32)
            print(frame_points_.shape)
            frame_points = frame_points_[:, :3] # N, 3
            print(frame_points.shape)
            sem_labels = np.zeros((frame_points.shape[0], 1))
            
            ts = time.time()
            # model inference
            input_list, up_point_list = inference(net, frame_points, sem_labels, patch_num_point=10000, seed_num=500)
            te = time.time()
            print(f'Infer time: {(te - ts):.6f}s')
            
            # FPS to 2* point nums
            upsample_result_np = np.concatenate(up_point_list, axis=0)

            upsample_result_th = torch.unsqueeze(torch.from_numpy(upsample_result_np), dim=0).cuda() # 1xNx3  
            #upsample_result_th = torch.unsqueeze(torch.cat(up_point_list, 0), dim=0) # 1xNx3
            #print(upsample_result_th.shape)
            upsampled_p_fps_id = pn2_utils.furthest_point_sample(upsample_result_th.contiguous(), 2*frame_points.shape[0])
            upsample_result_fps = pn2_utils.gather_operation(upsample_result_th.permute(0, 2, 1).contiguous(), upsampled_p_fps_id)
            upsample_result_fps = upsample_result_fps.permute(0,2,1) # 1xNx3
            print(upsample_result_fps.shape)

            # save predict point cloud
            t0 = time.time()
            out_pcd = (upsample_result_fps.squeeze(0)).cpu().detach().numpy().astype(np.float32)
            t1 = time.time()
            print(f'Convert time: {(t1 - t0):.6f}s')
            #out_pcd = (out_pcd*furthest_distance) + centroid
            np.savetxt(join(xyzresult_folder, basename(xyz_file)[:-4]+'_sr64.xyz'), out_pcd, fmt='%.6f')
            #np.savetxt(join(ori_xyz_folder, basename(xyz_file)[:-4]+'.xyz'), frame_points, fmt='%.6f')
            

            # very very important to use .astype('float32')!!
            # or the result is very likely saved wrongly
            out_file = join(xyzresult_folder.replace('_xyz', ''), basename(xyz_file)[:-4]+'.bin')
            out_pcd.astype('float32').tofile(out_file)

            cnt += 1
            if cnt >= max_cnt:
                break




if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id',help='Specify the index of the cuda device, e.g. 0, 1 ,2',default=0, type=int)
    parser.add_argument('--num_point', type=int, default=256,help='Point Number')
    parser.add_argument('--gt_num_point', type=int, default=1024,help='Point Number of GT points')
    parser.add_argument('--training_up_ratio', type=int, default=4,help='The Upsampling Ratio during training') 
    parser.add_argument('--testing_up_ratio', type=int, default=4, help='The Upsampling Ratio during testing')  
    parser.add_argument('--over_sampling_scale', type=float, default=1.5, help='The scale for over-sampling')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',help='Dimension of embeddings')
    parser.add_argument('--testing_anchor_num', type=int, default=114, metavar='N',help='The number of patches on the testing models')
    parser.add_argument('--pe_out_L', type=int, default=5, metavar='N',help='The parameter L in the position code')
    parser.add_argument('--feature_unfolding_nei_num', type=int, default=4, metavar='N',help='The number of neighbour points used while feature unfolding')
    parser.add_argument('--repulsion_nei_num', type=int, default=5, metavar='N',help='The number of neighbour points used in repulsion loss')

    # for phase train
    parser.add_argument('--batchsize', type=int, default=8, help='Batch Size during training')
    parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--if_bn', type=int, default=0, help='If using batch normalization')
    parser.add_argument('--neighbor_k', type=int, default=10, help='The number of neighbour points used in DGCNN')
    parser.add_argument('--mlp_fitting_str', type=str, default='256 128 64', metavar='None',help='mlp layers of the part surface fitting (default: None)')
    parser.add_argument('--mlp_projecting_str', type=str, default='None', metavar='None',help='mlp layers of the part surface projecting (default: None)')
    parser.add_argument('--glue_neighbor', type=int, default=4, help='The number of neighbour points used in glue process')
    parser.add_argument('--proj_neighbor', type=int, default=4, help='The number of neighbour points used in projection process')

    parser.add_argument('--weight_decay',default=0.00005, type=float)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--clip', type=float, default=1.0)

    # control the using mode
    parser.add_argument('--if_fix_sample', type=int, default=0, help='whether to use fix sampling')
    parser.add_argument('--if_use_siren', type=int, default=0, help='whether to use siren activation function')

    # paths
    #parser.add_argument('--input_path', type=str, default='/media/1TB_SSD/Sem_Kitti/down_dataset_32/sequences/08/velodyne/',help='network input path')
    parser.add_argument('--input_path', type=str, default='/media/1TB_HDD/KITTI_3Ddet_selected/In_64/',help='network input path')
    parser.add_argument('--gt_path', type=str, default='/media/1TB_HDD/KITTI_3Ddet_selected/In_64/',help='SR gt(64 beam) path')
    #parser.add_argument('--lbl_path', type=str, default='/media/1TB_SSD/Sem_Kitti/down_dataset_32/sequences/06/labels/',help='instance label path')
    parser.add_argument('--lbl_path', type=str, default='/media/1TB_SSD/Sem_Kitti/down_dataset_32/sequences/08/labels/',help='instance label path')
    #parser.add_argument('--ckpt_path', type=str, default='result_pugcnstyle_balance/epoch_10_ckpt.pt',help='checkpoint path')
    parser.add_argument('--ckpt_path', type=str, default='result_inc_no_stuff/epoch_10_ckpt.pt',help='checkpoint path')
    #parser.add_argument('--save_path', type=str, default='/media/1TB_HDD/pugcn_style_balance/ins_result/',help='xyz saving path')
    parser.add_argument('--save_path', type=str, default='/media/1TB_HDD/KITTI_3Ddet_selected/out_128/',help='xyz saving path')
    parser.add_argument('--ori_xyz', type=str, default='',help='original frame xyz saving path')
    args = parser.parse_args()


    #input_folder = '/media/1TB_SSD/Sem_Kitti/dataset/sequences/06/velodyne/'
    input_folder = args.input_path
    gt_folder = args.gt_path
    lbl_path = args.lbl_path
    ckpt_path = args.ckpt_path
    xyzresult_folder = args.save_path
    if not os.path.exists(xyzresult_folder):
        os.mkdir(xyzresult_folder)

    checkpoint = torch.load(ckpt_path)
    net = Net_conpu_v7(args).cuda()
    net.load_state_dict(checkpoint)
    print('prepare dataset .....')

    # read files list
    #gt_path = sorted(glob(join(input_folder, '8192', '*.xyz')))
    #input_path = sorted(glob(join(input_folder, '2048', '*.xyz')))
    #input_path = sorted(glob(join(input_folder, '*.bin')))
    input_path = sorted(glob(join(input_folder, '*.xyz')))
    #print(input_path)
    #gt_path = sorted(glob(join(gt_folder, '*.bin')))
    gt_path = sorted(glob(join(gt_folder, '*.xyz')))
    #lbl_path = sorted(glob(join(lbl_path, '*.label')))
    #print(input_path)
    lbl_path = sorted(glob(join(lbl_path, '*.txt')))
    #lbl_path = sorted(glob(join(lbl_path, '*.npy')))
    #print(input_path)

    print('start inference...')
    # net inference
    #net_infer(net, gt_path, input_path, xyzresult_folder, lbl_path)
    net_infer(net, input_path=input_path, xyzresult_folder=xyzresult_folder, ori_xyz_folder=args.ori_xyz, lbl_path=None, gt_path=None)
    print('Done.')
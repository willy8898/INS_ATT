import os
from os.path import join, basename
import time
import sys
import argparse
import random
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

def inference(net, pc, lbl_info, patch_num_point=256, radius=5.5, 
              patch_num_ratio=3, seed_num=None, sample_points_list=None):
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
        #print(upsample_result_fps.shape)
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

def net_infer(net, input_list, xyzresult_folder):
    ## choose random 100 frame
    #s = [i for i in range(len(input_list))]
    #random.shuffle(s)
    #choose_frame = s[:100]
    #print(choose_frame)
    #print(len(input_list))
    #raise ValueError

    #for i in tqdm(choose_frame, total=len(choose_frame)):
    for xyz_file in tqdm(input_list, total=len(input_list)):
        #xyz_file = input_list[i]
        # read point cloud file
        frame_points_ = np.fromfile(xyz_file, dtype=np.float32).reshape(-1, 4)
        frame_points = frame_points_[:, :3] # N, 3
        print(frame_points.shape)
        # fake sem_info
        sem_labels = np.zeros((frame_points.shape[0], 1))

        ts = time.time()
        # model inference
        _, up_point_list = inference(net, frame_points, sem_labels, patch_num_point=10000, seed_num=250)
        te = time.time()
        print(f'Infer time: {(te - ts):.6f}s')
        # FPS to 4* point nums
        upsample_result_np = np.concatenate(up_point_list, axis=0)
        upsample_result_th = torch.unsqueeze(torch.from_numpy(upsample_result_np), dim=0).cuda() # 1xNx3  
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
        np.savetxt(join(xyzresult_folder, basename(xyz_file)[:-4]+'.xyz'), out_pcd, fmt='%.6f')
        np.savetxt(join(xyzresult_folder.replace('Ins_aux_result', 'Input'), basename(xyz_file)[:-4]+'.xyz'), frame_points, fmt='%.6f')

        '''
        # very very important to use .astype('float32')!!
        # or the result is very likely saved wrongly
        out_file = join(xyzresult_folder.replace('_xyz', ''), basename(xyz_file)[:-4]+'.bin')
        out_pcd.astype('float32').tofile(out_file)
        '''


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
    parser.add_argument('--input_path', type=str, default='/media/1TB_SSD/Sem_Kitti/down_dataset_32/sequences/',help='input path')
    parser.add_argument('--ckpt_path', type=str, default='result_inc_no_stuff/epoch_10_ckpt.pt',help='checkpoint path')
    parser.add_argument('--save_path', type=str, default='/media/1TB_SSD/all_seq_result/',help='xyz saving path')
    parser.add_argument('--seq', type=int, default=1, help='select sequence')
    parser.add_argument('--frame_path', type=str, default='/home/gpl/Documents/NeuralPoints-INS_aux/model/select_frame/',help='select frame txt')
    args = parser.parse_args()

    ## Load Model
    checkpoint = torch.load(args.ckpt_path)
    net = Net_conpu_v7(args).cuda()
    net.load_state_dict(checkpoint)

    #seqs = [str(i).zfill(2) for i in range(1, 11)] # 01 ~ 10
    #for seq in seqs:
    #    xyzresult_folder = join(args.save_path, seq)
    #    if not os.path.exists(xyzresult_folder):
    #        os.mkdir(xyzresult_folder)
    #for seq in seqs:
    seq = str(args.seq).zfill(2)
    result_folder = join(args.save_path, seq)
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)

    input_folder = join(args.input_path, seq, 'velodyne')
    xyzresult_folder = join(args.save_path, seq, 'Ins_aux_result')
    inputsave_folder = join(args.save_path, seq, 'Input')
    if not os.path.exists(xyzresult_folder):
        os.mkdir(xyzresult_folder)
    if not os.path.exists(inputsave_folder):
        os.mkdir(inputsave_folder)
    
    '''
    # File list
    ## use select frame
    frame_file = join(args.frame_path, 'seq_'+seq+'.txt')
    with open(frame_file, 'r') as f:
        frames = f.readlines()
        frames = [i.rstrip() for i in frames]
    #print(frames)
    
    input_list = []
    #a = 0
    for frame in frames:
        file_path = join(input_folder, str(frame).zfill(6)+'.bin')
        input_list.append(file_path)
        #if a == 0:
        #    break
    #print(input_list)
    '''
    input_list = sorted(glob(join(input_folder, '*.bin')))
    # net inference
    net_infer(net, input_list, xyzresult_folder)

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
###### ------ ######
from network_ins_aux import Net_conpu_v7
from pointnet2 import pointnet2_utils as pn2_utils
###### ------ ######
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import datasets
import matplotlib.pyplot as plt

# replace Kmean to mean shift or its variants

# our ins embedding shape:
# thisbatchsize, self.args.num_point, self.emb_dims*2

# steps:
# 1. predict ins. embedding
# 2. clustering
# 3. mapping to point and assign color
# 4. visualize

"""
Shape:
- Input: `(N, C_{in}, H_{in}, W_{in})`
- Output:
    - Semantic Seg: `(N, N_{class}, H_{in}, W_{in})`
    - Instance Seg: `(N, 32, H_{in}, W_{in})`
    - Instance Cnt: `(N, 1)`
"""
def cluster(ins_seg_prediction):
    #embeddings = ins_seg_prediction.cpu()
    #embeddings = embeddings.numpy() # h, w, c
    embeddings = ins_seg_prediction

    try:
        # estimate bandwidth
        bandwidth = estimate_bandwidth(embeddings, quantile=0.1, n_samples=3000)
        #print('bandwidth: ', bandwidth)
        #Mean Shift method
        model = MeanShift(bandwidth=bandwidth, bin_seeding=True, max_iter=1000)
        model.fit(embeddings)
        labels = model.labels_
        cluster_centers = model.cluster_centers_
        #labels = model.fit_predict(embeddings)
    except ValueError as msg:
        print(msg)
        labels = np.zeros((embeddings.shape[0]))
        cluster_centers = np.zeros((embeddings.shape[0]))


    '''
    labels = KMeans(n_clusters=n_objects_prediction,
                    n_init=35, max_iter=500,
                    n_jobs=.n_workers).fit_predict(embeddings)
    instance_mask = np.zeros((seg_height, seg_width), dtype=np.uint8)

    fg_coords = np.where(sem_seg_prediction != 0)
    for si in range(len(fg_coords[0])):
        y_coord = fg_coords[0][si]
        x_coord = fg_coords[1][si]
        _label = labels[si] + 1
        instance_mask[y_coord, x_coord] = _label
    '''
    return labels, cluster_centers


# kitti sem label mapping
with open('semantic-kitti.yaml', 'r') as stream:
    doc = yaml.safe_load(stream)
    learning_map = doc['learning_map']
    learning_map_ = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
    for k, v in learning_map.items():
        learning_map_[k] = v

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

def inference(net, pc, lbl_info, patch_num_point=256, patch_num_ratio=3, radius=5.5, seed_num=None, frame=None):
    """
    pc: [P, 3]
    lbl_info: zero semantic lbl, no use
    """
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
    print("number of patches: %d" % upsample_result_fps.shape[0])

    patches, instances = extract_knn_patch(upsample_result_fps, lbls, pc, patch_num_point)
    centers = upsample_result_fps
    #print(patches.shape)
    #print(instances.shape)

    input_list = []
    up_point_list = []
    net.eval()
    no_infer_num = 0
    cnt = 0
    with torch.no_grad():
        for i in range(len(patches)):
            patch, ins = patches[i, :, :3], instances[i, :, :1]
            #print(patch.shape)
            #print(ins.shape)
            mask = np.sum(np.square(patch[:, :3] - centers[i:i+1, :3]), axis=1) < radius ** 2
            mask_inds = np.where(mask)[0].astype(np.int32)
            patch = patch[mask_inds, :3]
            ins = ins[mask_inds, :1]
            sem_fake = np.zeros_like((ins))
            #print(patch.shape)
            #print(ins.shape)
            #print(f'max_x: {np.max(patch, axis=0)[0]}, max_y: {np.max(patch, axis=0)[1]}, max_z: {np.max(patch, axis=0)[2]}')
            #print(f'min_x: {np.min(patch, axis=0)[0]}, min_y: {np.min(patch, axis=0)[1]}, min_z: {np.min(patch, axis=0)[2]}')

            p = torch.from_numpy(np.expand_dims(patch, axis=0)).cuda() # [bs, 256, 3]
            #i = torch.from_numpy(np.expand_dims(ins, axis=0)).type(torch.FloatTensor).cuda() # [bs, 256, 1]
            i = torch.from_numpy(np.expand_dims(sem_fake, axis=0)).type(torch.FloatTensor).cuda() # [bs, 256, 1]
            #print(p.shape)
            #print(i.shape)
            
            if p.shape[1] < 20:
                up_point_list.append(p.squeeze(0).cpu().detach().numpy().astype(np.float32))
                no_infer_num += 1
            else:
                # predict
                # ins_embed: thisbatchsize, self.args.num_point, self.emb_dims*2
                upsampled_p, ins_embed = net(p, i, is_train=True)
                #print(upsampled_p.shape)
                #print(ins_embed.shape)
                upsample_point_np = upsampled_p.squeeze(0).cpu().detach().numpy().astype(np.float32)

                labels, cluster_centers = cluster(ins_embed.squeeze(0).cpu().detach().numpy().astype(np.float32))
                #print(np.unique(labels, return_counts=True))
                #print(labels.shape)

                ins_patch = np.concatenate((patch, labels.reshape(patch.shape[0], 1)), axis=1)
                ins_patch_gt = np.concatenate((patch, ins.reshape(patch.shape[0], 1)), axis=1)
                
                # save patches
                #if upsample_point_np.shape[0] > 200:
                    #np.savetxt(join("/media/1TB_HDD/ins_seg_test/", str(cnt)+'_sr.xyz'), upsample_point_np, fmt='%.6f')
                    #np.savetxt(join("/media/1TB_HDD/ins_seg_test/up_ins/", str(frame)+'_'+str(cnt)+'_ins.xyz'), ins_patch, fmt='%.6f')
                    #np.savetxt(join("/media/1TB_HDD/ins_seg_test/up_ins/", str(frame)+'_'+str(cnt)+'_insgt.xyz'), ins_patch_gt, fmt='%.6f')
                    #np.savetxt(join("/media/1TB_HDD/ins_seg_test/", str(cnt)+'_in.xyz'), patch, fmt='%.6f')
                
                np.savetxt(join("/media/1TB_HDD/ins_seg_test/up_ins/", str(frame)+'_'+str(cnt)+'_ins.xyz'), ins_patch, fmt='%.6f')
                np.savetxt(join("/media/1TB_HDD/ins_seg_test/up_ins/", str(frame)+'_'+str(cnt)+'_insgt.xyz'), ins_patch_gt, fmt='%.6f')
                cnt += 1
                #up_point_list.append(upsampled_p.squeeze(0))
                up_point_list.append(upsample_point_np)
            
            #if cnt > 100:
            #raise ValueError('A very specific bad thing happened.')
    print(' no_infer_num: ',  no_infer_num)
    return input_list, up_point_list

def net_infer(net, gt_path, input_path, xyzresult_folder, lbl_path, chfloss_txt_path='./chf_d.txt'):
    #cnt = 0
    #max_cnt = 100
    for xyz_file, gt_xyz_file, lbl_file in tqdm(zip(input_path, gt_path, lbl_path), total=len(gt_path)):
        #cnt += 1
        #if cnt != 48:
        #    continue
        # read point cloud file
        frame_id = basename(xyz_file).replace('.bin', '')
        frame_points_ = np.fromfile(xyz_file, dtype=np.float32).reshape((-1, 4))
        frame_points = frame_points_[:, :3] # N, 3
        print(frame_points.shape)
        gt_points = np.fromfile(gt_xyz_file, dtype=np.float32).reshape((-1, 4))
        gt_points = gt_points[:, :3]
        gt_points = torch.from_numpy(gt_points).cuda()
        # read instance info
        #sem_labels = np.zeros((frame_points.shape[0], 1))
        lbl_info = np.fromfile(lbl_file, dtype=np.int32).reshape((-1, 1))
        sem_labels = lbl_info
        #sem_labels = lbl_info & 0xFFFF  # semantic label in lower half
        #sem_labels = learning_map_[sem_labels]

        ts = time.time()
        # model inference
        input_list, up_point_list = inference(net, frame_points, sem_labels, patch_num_point=10000, seed_num=250, frame=frame_id)
        te = time.time()
        print(f'Infer time: {(te - ts):.6f}s')
        #upsample_result_np = np.concatenate(up_point_list, axis=0)

        #raise ValueError('A very specific bad thing happened.')
        '''
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
        np.savetxt(join(xyzresult_folder, basename(xyz_file)[:-4]+'.xyz'), out_pcd, fmt='%.6f')
        '''
        #cnt += 1
        #if cnt >= max_cnt:
        #    break





if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    #parser.add_argument('--device_id',help='Specify the index of the cuda device, e.g. 0, 1 ,2',default=0, type=int)
    parser.add_argument('--num_point', type=int, default=256,help='Point Number (No use)')
    parser.add_argument('--training_up_ratio', type=int, default=4,help='The Upsampling Ratio during training') 
    parser.add_argument('--testing_up_ratio', type=int, default=4, help='The Upsampling Ratio during testing')  
    parser.add_argument('--over_sampling_scale', type=float, default=1.5, help='The scale for over-sampling')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',help='Dimension of embeddings')
    parser.add_argument('--pe_out_L', type=int, default=5, metavar='N',help='The parameter L in the position code')
    parser.add_argument('--feature_unfolding_nei_num', type=int, default=4, metavar='N',help='The number of neighbour points used while feature unfolding')

    # for phase train
    #parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--if_bn', type=int, default=0, help='If using batch normalization')
    parser.add_argument('--neighbor_k', type=int, default=10, help='The number of neighbour points used in DGCNN')
    parser.add_argument('--mlp_fitting_str', type=str, default='256 128 64', metavar='None',help='mlp layers of the part surface fitting (default: None)')
    #parser.add_argument('--mlp_projecting_str', type=str, default='None', metavar='None',help='mlp layers of the part surface projecting (default: None)')
    #parser.add_argument('--glue_neighbor', type=int, default=4, help='The number of neighbour points used in glue process')
    #parser.add_argument('--proj_neighbor', type=int, default=4, help='The number of neighbour points used in projection process')

    #parser.add_argument('--weight_decay',default=0.00005, type=float)
    #parser.add_argument('--epsilon', type=float, default=1e-8)
    #parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    #parser.add_argument('--clip', type=float, default=1.0)

    # control the using mode
    parser.add_argument('--if_fix_sample', type=int, default=0, help='whether to use fix sampling')
    parser.add_argument('--if_use_siren', type=int, default=0, help='whether to use siren activation function')

    # paths
    #parser.add_argument('--input_path', type=str, default='/media/1TB_SSD/Sem_Kitti/down_dataset_32/sequences/06/velodyne/',help='network input path')
    parser.add_argument('--input_path', type=str, default='/media/1TB_SSD/Sem_Kitti/down_dataset_32/sequences/',help='input path')
    parser.add_argument('--gt_path', type=str, default='/media/1TB_SSD/Sem_Kitti/dataset/sequences/',help='SR gt(64 beam) path')
    parser.add_argument('--lbl_path', type=str, default='/media/1TB_SSD/Sem_Kitti/down_dataset_32/sequences/',help='instance label path')
    parser.add_argument('--ckpt_path', type=str, default='result_inc_no_stuff/epoch_10_ckpt.pt',help='checkpoint path')
    parser.add_argument('--save_path', type=str, default='/media/1TB_HDD/ins_seg_test/up_ins/',help='xyz saving path')
    parser.add_argument('--seq', type=int, default=6, help='select sequence')
    parser.add_argument('--frame_path', type=str, default='/home/gpl/Documents/NeuralPoints-INS_aux/model/select_frame/',help='select frame txt')
    args = parser.parse_args()

    
    ckpt_path = args.ckpt_path
    xyzresult_folder = args.save_path
    if not os.path.exists(xyzresult_folder):
        os.mkdir(xyzresult_folder)

    checkpoint = torch.load(ckpt_path)
    net = Net_conpu_v7(args).cuda()
    net.load_state_dict(checkpoint)
    print('prepare dataset .....')
    
    # File list
    seq = str(args.seq).zfill(2)
    input_folder = join(args.input_path, seq, 'velodyne')
    gt_folder = join(args.gt_path, seq, 'velodyne')
    lbl_folder = join(args.lbl_path, seq, 'labels')
    ## use select frame
    frame_file = join(args.frame_path, 'seq_'+seq+'.txt')
    with open(frame_file, 'r') as f:
        frames = f.readlines()
        frames = [i.rstrip() for i in frames]
    #print(frames)
    
    input_path = []
    gt_path = []
    lbl_path = []
    #a = 0
    for frame in frames:
        file_path = join(input_folder, str(frame).zfill(6)+'.bin')
        input_path.append(file_path)
        gt_file_path = join(gt_folder, str(frame).zfill(6)+'.bin')
        gt_path.append(gt_file_path)
        lbl_file_path = join(lbl_folder, str(frame).zfill(6)+'.label')
        lbl_path.append(lbl_file_path)
        #if a == 0:
        #    break
    #print(input_list)

    '''
    #input_folder = args.input_path
    #gt_folder = args.gt_path
    #lbl_path = args.lbl_path
    # read files list
    input_path = sorted(glob(join(input_folder, '*.bin')))
    gt_path = sorted(glob(join(gt_folder, '*.bin')))
    lbl_path = sorted(glob(join(lbl_path, '*.label')))
    '''

    print('start inference...')
    # net inference
    net_infer(net, gt_path, input_path, xyzresult_folder, lbl_path)
    print('Done.')


    """
    #create datasets
    X, _ = datasets.make_blobs(n_samples=50, centers=3, n_features=2, random_state= 20, cluster_std = 1.5)
    print(X.shape)

    ins_labels, cluster_centers = cluster(X)
    
    #results visualization
    my_dpi = 96
    plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
    plt.scatter(X[:,0], X[:,1], c = ins_labels)
    plt.axis('equal')
    plt.title('Prediction')
    #plt.show()
    plt.savefig('my_fig.png', dpi=my_dpi)
    """


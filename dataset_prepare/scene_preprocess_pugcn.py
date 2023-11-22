# OP: random extract patches from kitti scene(and also balanced with classes), 
# then save them in npy files
import time
import os
from os.path import join
from os import listdir
from turtle import down
import yaml
import numpy as np
import torch
from sklearn.neighbors import KDTree, NearestNeighbors
from tqdm import tqdm
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../code/')
from pointnet2 import pointnet2_utils as pn2_utils
import open3d


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
with open('semantic-kitti.yaml', 'r') as stream:
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

def extract_knn_patch(queries, pc, k, sem_label, ins_pred):
    """
    queries [M, C]
    pc [P, C]
    """
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pc)
    knn_idx = knn_search.kneighbors(queries, return_distance=False)
    k_patches = np.take(pc, knn_idx, axis=0)  # M, K, C
    k_semlabel = np.take(sem_label, knn_idx, axis=0)  # M, K, C
    k_inspred = np.take(ins_pred, knn_idx, axis=0) # M, K, C
    return k_patches, k_semlabel, k_inspred

def seed_sampling(pc, patch_num_point=256, patch_num_ratio=3):
    # FPS to get patch
    seed1_num = int(pc.shape[0] / patch_num_point * patch_num_ratio)
    
    points = torch.from_numpy(np.expand_dims(pc, axis=0)).cuda() # 1, P, 3
    upsampled_p_fps_id = pn2_utils.furthest_point_sample(points.contiguous(), seed1_num)
    upsample_result_fps = pn2_utils.gather_operation(points.permute(0, 2, 1).contiguous(), upsampled_p_fps_id)
    upsample_result_fps = (upsample_result_fps.permute(0,2,1).squeeze(0)).cpu().detach().numpy().astype(np.float32)
    print("number of patches: %d" % upsample_result_fps.shape[0])

    patches = extract_knn_patch(upsample_result_fps, pc, patch_num_point)


if __name__ == '__main__':
    #np.random.seed(0)
    balance_classes = True
    dense_path = '/media/1TB_SSD/Sem_Kitti/dataset/'
    path = '/media/1TB_SSD/Sem_Kitti/down_dataset_32/'
    #save_path = '/media/1TB_HDD/kitti_ins_balance/'
    save_path = '/media/1TB_HDD/pugcn_style_256/'
    in_R = 3.5 # meter

    sequences = ['{:02d}'.format(i) for i in range(11)] # 0 ~ 10

    # List of all files in each sequence
    frames = []
    for seq in sequences:
        velo_path = join(path, 'sequences', seq, 'velodyne')
        frames_ = np.sort([vf[:-4] for vf in listdir(velo_path) if vf.endswith('.bin')])
        frames.append(frames_.tolist())

    # 1: car, 2: bicycle, 3: motorcycle
    # 4: truck, 5: other-vehicle, 6: person
    # 7: bicyclist, 8: motorcyclist
    total_class = [i for i in range(20)]
    ins_class = [1, 2, 3, 4, 5, 6, 7, 8]
    stuff_class = [0, 9, 10, 11, 12, 13, 14,
                   15, 16, 17, 18, 19]
    
    # thing class
    save_cnt = 0
    for seq, frame in zip(sequences, frames):
        dense_seq_path = join(dense_path, 'sequences', seq)
        seq_path = join(path, 'sequences', seq)
        print(f'Processing seqence {seq} of {len(frame)} frames....')

        sub_dir = ['subscene_xyz', 'subscene', 'label', 'label_origin', 'subscene_dense', 'label_dense']
        for sub in sub_dir:
            directory = join(save_path, sub, seq)
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        for f_i in tqdm(frame, total=len(frame)):
            dense_velo_file = join(dense_seq_path, 'velodyne', f_i + '.bin')
            dense_label_file = join(dense_seq_path, 'labels', f_i + '.label')
            velo_file = join(seq_path, 'velodyne', f_i + '.bin')
            label_file = join(seq_path, 'labels', f_i + '.label')  

            dense_frame_points = np.fromfile(dense_velo_file, dtype=np.float32).reshape((-1, 4))[:, :3]
            dense_frame_labels = np.fromfile(dense_label_file, dtype=np.int32).reshape((-1, 1))
            frame_points = np.fromfile(velo_file, dtype=np.float32).reshape((-1, 4))[:, :3]
            # focus on sparse lbl to select interested region
            frame_labels = np.fromfile(label_file, dtype=np.int32).reshape((-1, 1))
            sem_labels_ = frame_labels & 0xFFFF  # semantic label in lower half
            sem_labels = learning_map[sem_labels_]

            # TO-DO: filter the thing class
            for wanted_label in ins_class: 
                wanted_mask = np.where(sem_labels == wanted_label)[0]
                wanted_ins_lbl = frame_labels[wanted_mask, :1]
                ins_cls, ins_counts = np.unique(wanted_ins_lbl ,return_counts=True)

                for ins_i, ins_cnt_i in zip(ins_cls, ins_counts):
                    ins_center_idx = np.random.choice(np.where(frame_labels == ins_i)[0])
                    p0 = frame_points[ins_center_idx:ins_center_idx+1, :3]
                    # Eliminate points further than config.in_radius
                    dense_mask = np.sum(np.square(dense_frame_points[:, :3] - p0), axis=1) < in_R ** 2
                    dense_mask_inds = np.where(dense_mask)[0].astype(np.int32)
                    mask = np.sum(np.square(frame_points[:, :3] - p0), axis=1) < in_R ** 2
                    mask_inds = np.where(mask)[0].astype(np.int32)
                    #print(mask_inds.shape)

                    # resample to 256(input) and 512(gt)
                    if len(mask_inds) > 512 or len(mask_inds) < 256: # skip if too many or less points
                        continue
                    idx = np.random.choice(mask_inds.shape[0], 256, replace=False)
                    mask_inds = mask_inds[idx]
                    if dense_mask_inds.shape[0] < 512: # skip if too many or less points
                        continue
                    dense_idx = np.random.choice(dense_mask_inds.shape[0], 512, replace=False)
                    dense_mask_inds = dense_mask_inds[dense_idx]

                    dense_new_points = dense_frame_points[dense_mask_inds, :3]
                    dense_new_label = dense_frame_labels[dense_mask_inds, :1]
                    # shuffle order
                    rand_order = np.random.permutation(mask_inds)
                    new_points = frame_points[rand_order, :3]
                    ins_labels_shuffle = frame_labels[rand_order]
                    ori_sem = sem_labels_[rand_order]
                    #print(np.unique(ori_sem, return_counts=True))
                    #sem_labels_shuffle = sem_labels[rand_order]

                    # filter not thing class points as 0
                    has_ins = False
                    not_thing_mask = np.ones((ins_labels_shuffle.shape[0], ), dtype=bool)
                    for things_id in things_ids:
                        is_thing = np.where(ori_sem == things_id)[0]
                        not_thing_mask[is_thing] = False
                        if is_thing.shape[0] > 5: # check if scene has instance
                            has_ins = True
                    not_thing_mask = np.where(not_thing_mask)[0]
                    ins_labels_final = ins_labels_shuffle.copy()
                    ins_labels_final[not_thing_mask] = 0
                    #print(np.unique(ins_labels_final, return_counts=True))
                    #print((ins_labels_final == ins_labels_shuffle).all())

                    # save the data
                    if ins_cnt_i > 1 and has_ins: # save the subscene if ins_num > 1 and its points should be more than one
                        #print('saving....')
                        #print(ins_cls)
                        #print(ins_counts)
                        #print(mask_inds.shape)
                        #print(dense_mask_inds.shape)
                        point_cloud = open3d.geometry.PointCloud()
                        point_cloud.points = open3d.utility.Vector3dVector(new_points.squeeze())
                        open3d.io.write_point_cloud(join(save_path, 'subscene_xyz', seq, f'{seq}_{f_i}_{wanted_label}_{ins_i}.xyz'), point_cloud)
                        save_cnt += 1
                        point_cloud.points = open3d.utility.Vector3dVector(dense_new_points)
                        open3d.io.write_point_cloud(join(save_path, 'subscene_xyz', seq, f'{seq}_{f_i}_{wanted_label}_{ins_i}_dense.xyz'), point_cloud)

                        
                        #print(ins_labels_shuffle.shape)
                        #print(new_points.shape)
                        #print(dense_new_label.shape)
                        #print(dense_new_points.shape)
                        
                        np.save(join(save_path, 'subscene', seq, f'{seq}_{f_i}_{wanted_label}_{ins_i}_scene.npy'), new_points)
                        np.save(join(save_path, 'label', seq, f'{seq}_{f_i}_{wanted_label}_{ins_i}_label.npy'), ins_labels_final)
                        np.save(join(save_path, 'label_origin', seq, f'{seq}_{f_i}_{wanted_label}_{ins_i}_label_origin.npy'), ins_labels_shuffle)
                        np.save(join(save_path, 'subscene_dense', seq, f'{seq}_{f_i}_{wanted_label}_{ins_i}_dense.npy'), dense_new_points)
                        np.save(join(save_path, 'label_dense', seq, f'{seq}_{f_i}_{wanted_label}_{ins_i}_dense.npy'), dense_new_label)

                #raise ValueError('Stop')

        print(f'seqence {seq} has save {save_cnt} subscene..')
    

    '''
    # stuff class
    stuff_cnt = 0
    for seq, frame in zip(sequences, frames):
        dense_seq_path = join(dense_path, 'sequences', seq)
        seq_path = join(path, 'sequences', seq)
        print(f'Processing seqence {seq} of {len(frame)} frames....')

        sub_dir = ['subscene_xyz_stuff', 'subscene_stuff', 'label_stuff', 'label_origin_stuff', 'subscene_dense_stuff', 'label_dense_stuff']
        for sub in sub_dir:
            directory = join(save_path, sub, seq)
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        for f_i in tqdm(frame, total=len(frame)):
            dense_velo_file = join(dense_seq_path, 'velodyne', f_i + '.bin')
            dense_label_file = join(dense_seq_path, 'labels', f_i + '.label')
            velo_file = join(seq_path, 'velodyne', f_i + '.bin')
            label_file = join(seq_path, 'labels', f_i + '.label')  

            dense_frame_points = np.fromfile(dense_velo_file, dtype=np.float32).reshape((-1, 4))[:, :3]
            dense_frame_labels = np.fromfile(dense_label_file, dtype=np.int32).reshape((-1, 1))
            frame_points = np.fromfile(velo_file, dtype=np.float32).reshape((-1, 4))[:, :3]
            # focus on sparse lbl to select interested region
            frame_labels = np.fromfile(label_file, dtype=np.int32).reshape((-1, 1))
            sem_labels_ = frame_labels & 0xFFFF  # semantic label in lower half
            sem_labels = learning_map[sem_labels_]

            try_cnt = 0
            for wanted_label in stuff_class:
                wanted_mask = np.where(sem_labels == wanted_label)[0]
                if len(wanted_mask) == 0: # skip if no this class
                    continue

                center_idx = np.random.choice(wanted_mask)
                p0 = frame_points[center_idx:center_idx+1, :3]
                # Eliminate points further than config.in_radius
                dense_mask = np.sum(np.square(dense_frame_points[:, :3] - p0), axis=1) < in_R ** 2
                dense_mask_inds = np.where(dense_mask)[0].astype(np.int32)
                mask = np.sum(np.square(frame_points[:, :3] - p0), axis=1) < in_R ** 2
                mask_inds = np.where(mask)[0].astype(np.int32)
                #print(mask_inds.shape)

                if len(mask_inds) > 10000 or len(mask_inds) < 20: # skip if too many or less points
                    continue

                dense_new_points = dense_frame_points[dense_mask_inds, :3]
                dense_new_label = dense_frame_labels[dense_mask_inds, :1]
                # shuffle order
                rand_order = np.random.permutation(mask_inds)
                new_points = frame_points[rand_order, :3]
                ins_labels_shuffle = frame_labels[rand_order]
                ori_sem = sem_labels_[rand_order]

                # filter not thing class points as 0
                not_thing_mask = np.ones((ins_labels_shuffle.shape[0], ), dtype=bool)
                for things_id in things_ids:
                    is_thing = np.where(ori_sem == things_id)[0]
                    not_thing_mask[is_thing] = False
                not_thing_mask = np.where(not_thing_mask)[0]
                ins_labels_final = ins_labels_shuffle.copy()
                ins_labels_final[not_thing_mask] = 0

                # save the subscene
                point_cloud = open3d.geometry.PointCloud()
                point_cloud.points = open3d.utility.Vector3dVector(new_points.squeeze())
                open3d.io.write_point_cloud(join(save_path, 'subscene_xyz_stuff', seq, f'{seq}_{f_i}_{wanted_label}.xyz'), point_cloud)
                stuff_cnt += 1
                point_cloud.points = open3d.utility.Vector3dVector(dense_new_points)
                open3d.io.write_point_cloud(join(save_path, 'subscene_xyz_stuff', seq, f'{seq}_{f_i}_{wanted_label}_dense.xyz'), point_cloud)

                """
                print(ins_labels_shuffle.shape)
                print(new_points.shape)
                print(dense_new_label.shape)
                print(dense_new_points.shape)
                """
                np.save(join(save_path, 'subscene_stuff', seq, f'{seq}_{f_i}_{wanted_label}_scene.npy'), new_points)
                np.save(join(save_path, 'label_stuff', seq, f'{seq}_{f_i}_{wanted_label}_label.npy'), ins_labels_final)
                np.save(join(save_path, 'label_origin_stuff', seq, f'{seq}_{f_i}_{wanted_label}_label_origin.npy'), ins_labels_shuffle)
                np.save(join(save_path, 'subscene_dense_stuff', seq, f'{seq}_{f_i}_{wanted_label}_dense.npy'), dense_new_points)
                np.save(join(save_path, 'label_dense_stuff', seq, f'{seq}_{f_i}_{wanted_label}_dense.npy'), dense_new_label)

        print(f'seqence {seq} has save {stuff_cnt} subscene..')

        '''
                




            




from glob import glob
from os.path import join, basename
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KDTree, NearestNeighbors


# https://github.com/VincentCheungM/Run_based_segmentation/issues/3
'''
KITTI data are sorted. 
The points are recorded in an orderly manner, the points of one ring follow the points of another in the direction of laser rotation. 
Rings can be separated by tracing the change of the quadrant
'''
def get_quadrant(point):
        res = 0
        x = point[0]
        y = point[1]
        if x > 0 and y >= 0:
            res = 1
        elif x <= 0 and y > 0:
            res = 2
        elif x < 0 and y <= 0:
            res = 3
        elif x >= 0 and y < 0:
            res = 4
        return res

def add_ring_info(scan_points):
    num_of_points = scan_points.shape[0]
    scan_points = np.hstack([scan_points,
                                np.zeros((num_of_points, 1))])
    velodyne_rings_count = 64
    previous_quadrant = 0
    ring = 0
    for num in range(num_of_points-1, -1, -1):
        quadrant = get_quadrant(scan_points[num])
        if quadrant == 4 and previous_quadrant == 1 and ring < velodyne_rings_count-1:
            ring += 1
        scan_points[num, 4] = ring
        previous_quadrant = quadrant
    return scan_points

def extract_knn_patch(queries, pc, k):
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
    return k_patches, knn_idx

if __name__ == "__main__":
    velo_dir = '/media/1TB_SSD/Sem_Kitti/dataset/sequences/'
    #SR_dir = '/media/4TB_HDD/Ins_aux_result/no_stuff_epoch_10_ckpt_use_pred/'
    #save_path = '/media/4TB_HDD/Ins_aux_result/no_stuff_epoch_10_ckpt_use_pred_f/'
    SR_dir = '/media/1TB_HDD/kitti_trainset/kitti_ins_balance_result/no_stuff_epoch_10_ckpt/'
    save_path = '/media/1TB_HDD/comparsion_result/kitti_left_32/'
    # for train and val seqs (with labels)
    #seq_list = ['00', '01', '02', '03', '04',
    #            '05', '06', '07', '08', '09',
    #            '10']
    seq_list = ['08']
    file_list = []
    for seq in seq_list:
        seq_files = sorted(glob(join(velo_dir, seq, 'velodyne', '0000*.bin')))
        file_list.extend(seq_files)
    
    sr_file_list = []
    for seq in seq_list:
        seq_files = sorted(glob(join(SR_dir, '0000*.xyz')))
        sr_file_list.extend(seq_files)
    
    for velo_filename, sr_filename in tqdm(zip(file_list, sr_file_list), total=len(sr_file_list)):
        """
        # pitch(radian): ring_idx +- 0.004
        
        ## 1. calculate GT pitch mean
        ## 2. calculate SR point pitch
        ## 3. find GT nearest neighbor of pitch for each SR points -> assign ring idx
        ## 4. find not input ring_idx: (ring idx + 1) % 2 ==0

        scan_points = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
        scan_points = add_ring_info(scan_points)
        ring_idx = scan_points[:, 4]
        
        ## 1. calculate GT pitch mean
        total_ring_idx = 64
        ring_idx_mean = np.zeros((total_ring_idx), dtype=np.float32)
        for idx in range(total_ring_idx):
            ring_mask = np.where(ring_idx == idx)[0]
            ring_pts = scan_points[ring_mask, :3]

            depth = np.linalg.norm(ring_pts, 2, axis=1)
            pitch = np.arcsin(ring_pts[:, 2] / depth) # arcsin(z, depth)
            ring_idx_mean[idx] = pitch.mean()
        # print(ring_idx_mean)
        
        ## 2. calculate SR point pitch
        SR_points = np.loadtxt(sr_filename)
        sr_depth = np.linalg.norm(SR_points, 2, axis=1)
        sr_pitch = np.arcsin(SR_points[:, 2] / sr_depth) # arcsin(z, depth)
        ## 3. find GT nearest neighbor of pitch for each SR points -> assign ring idx
        _, knn_idx = extract_knn_patch(queries=sr_pitch.reshape(-1, 1), pc=ring_idx_mean.reshape(-1, 1), k=1)
        ## 4. find not input ring_idx: (ring idx + 1) % 2 ==0
        knn_idx = knn_idx.reshape(-1)
        non_input_point_mask = np.where((knn_idx+1) % 2 == 0)[0]
        non_input_point = SR_points[non_input_point_mask, :3]

        #np.savetxt(join(save_path, basename(velo_filename)[:-4]+'_32.xyz'), down_points, fmt='%.6f')
        np.savetxt(join(save_path, basename(velo_filename)[:-4]+'.xyz'), non_input_point, fmt='%.6f')
        """

        # find left 32 beam 
        scan_points = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
        scan_points = add_ring_info(scan_points)
        ring_idx = scan_points[:, 4]

        left_32_mask = np.where((ring_idx+1) % 2 == 0)[0]
        left_32 = scan_points[left_32_mask, :3]

        np.savetxt(join(save_path, basename(velo_filename)[:-4]+'.xyz'), left_32, fmt='%.6f')
        

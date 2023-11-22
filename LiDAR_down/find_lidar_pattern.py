import argparse
import numpy as np
import os
import os.path as osp
from os.path import join, basename
from glob import glob
import cv2
from tqdm import tqdm


def save_to_img(pcd, write_path):
    norm_image = cv2.normalize(pcd, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite(write_path, norm_image*255)


def filter_by_ringidx(file_path, write_path, scan_lines=64):
    # read points
    #scan = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))  
    scan = np.loadtxt(file_path)
    x, y, z = scan[:, 0], scan[:, 1], scan[:, 2]
    #print('64 beams point shape: ', scan.shape)

    #########################################################
    # note: Here, make left bottom as (0, 0) 
    #########################################################
    # find row id (ring channel)
    # steps: 
    # 1. fit the points to the vertical continuous angles
    depth = np.linalg.norm(scan, 2, axis=1)
    pitch = np.arcsin(scan[:, 2] / depth) # arcsin(z, depth)
    # 2. normalize the vertical angles
    fov_down = -24.8 / 180.0 * np.pi
    fov = (abs(-24.8) + abs(2.0)) / 180.0 * np.pi
    proj_y = (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
    # 3. scale by given lidar scan lines number
    proj_y *= scan_lines  # in [0.0, H]
    # 4. Discrtize the angles
    proj_y = np.floor(proj_y)
    proj_y = np.minimum(scan_lines - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0, H-1]
    proj_y = proj_y.reshape(-1, 1)

    # 5. filter by downsample ratio
    down_mask = np.where(proj_y % 2 == 0)[0]
    down_scan = scan[down_mask, :3]
    #print('downsample point shape: ', down_scan.shape)

    ### save file
    #file_name = basename(file_path)
    #np.save(join(write_path, file_name), down_scan)
    file_name = basename(file_path).replace('.bin', '.xyz')
    np.savetxt(join(write_path, file_name), down_scan, fmt='%.6f')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", help="file folder", type=str,
                        default="/media/1TB_SSD/Sem_Kitti/dataset/sequences/")
    parser.add_argument("-w", "--write_path", help="write out folder", type=str,
                        default="/media/1TB_SSD/Sem_Kitti/testing_save/")
    parser.add_argument("-s", "--scan_line", help="lidar scan line number", type=int,
                        default=128)
    args = parser.parse_args()

    ###############################
    # Read data and main process
    ###############################
    print(args.data_path)
    data_path = args.data_path
    #seq_list = ['00', '01', '02', '03', '04', '05',
    #              '06', '07', '08', '09', '10']
    #seq_list = ['08']

    #file_list = []
    #for seq in seq_list:
    #    file_list.extend(sorted(glob(osp.join(data_path, seq, 'velodyne', '*.bin'))))
    #    #lbl_list = sorted(glob(osp.join(data_path, seq, 'labels', '*.label')))

    file_list = sorted(glob(join(data_path, '*.xyz')))

    for file_path in tqdm(file_list, total=len(file_list)):
        filter_by_ringidx(file_path, args.write_path, args.scan_line)
        #break

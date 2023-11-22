# https://github.com/TixiaoShan/LIO-SAM/blob/master/config/doc/kitti2bag/kitti2bag.py#L239
# OP: downsample semkitti pcd and label to (64 to 16 beam)
from cProfile import label
import os.path as osp
import os
import numpy as np
import open3d as o3d
from glob import glob
from tqdm import tqdm


velo_data_dir = '/media/1TB_SSD/Sem_Kitti/dataset/sequences/'
seq_list = ['00', '01', '02', '03', '04',
            '05', '06', '07', '08', '09',
            '10']
#seq_list = ['00']
output_data_dir = '/media/1TB_SSD/Sem_Kitti/down_dataset/'

file_list = []
label_list = []
for seq in seq_list:
    seq_files = sorted(glob(osp.join(velo_data_dir, seq, 'velodyne', '*.bin')))
    file_list.append(seq_files)
    seq_lbl = sorted(glob(osp.join(velo_data_dir, seq, 'labels', '*.label')))
    label_list.append(seq_lbl)

for file_seq, label_seq in zip(file_list, label_list):
    seq = (file_seq[0]).split('/')[-3]
    output_file_dir = osp.join(output_data_dir, seq, 'velodyne')
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    output_lbl_dir = osp.join(output_data_dir, seq, 'labels')
    if not os.path.exists(output_lbl_dir):
        os.makedirs(output_lbl_dir)

    for velo_filename, velo_label in tqdm(zip(file_seq, label_seq)):
        # read binary data
        scan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
        label = (np.fromfile(velo_label, dtype=np.uint32)).reshape(-1, 1)
        #print(scan.shape)
        #print(scan[0]) # [x, y, z, intensity]

        # get ring channel
        depth = np.linalg.norm(scan, 2, axis=1)
        pitch = np.arcsin(scan[:, 2] / depth) # arcsin(z, depth)
        #fov_down = -24.8 / 180.0 * np.pi
        fov_down = -25.0 / 180.0 * np.pi
        #fov = (abs(-24.8) + abs(2.0)) / 180.0 * np.pi
        fov = (abs(-25.0) + abs(3.0)) / 180.0 * np.pi
        proj_y = (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
        proj_y *= 64  # in [0.0, H]
        proj_y = np.floor(proj_y)
        proj_y = np.minimum(64 - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        proj_y = proj_y.reshape(-1, 1)
        scan = np.concatenate((scan,proj_y), axis=1)
        scan = scan.tolist()
        for i in range(len(scan)):
            scan[i][-1] = int(scan[i][-1])

        #print(scan[0]) # [x, y, z, intensity, ring channel]

        down_4_pcd = []
        down_4_lbl = []
        for i in range(len(scan)):
            if scan[i][-1] % 4 == 0:
                down_4_pcd.append(scan[i])
                down_4_lbl.append(label[i])

        down_4_pcd_np = np.array(down_4_pcd)
        down_4_lbl_np = np.array(down_4_lbl)
        down_pcd_xyzr = down_4_pcd_np[:, :4].flatten()
        #print(down_pcd_xyzr.shape)
        #print(down_4_lbl_np.shape)

        # very very important to use .astype('float32')!!
        # or the result is very likely saved wrongly
        out_filename = velo_filename.split('/')[-1]
        out_file = osp.join(output_file_dir, out_filename)
        down_pcd_xyzr.astype('float32').tofile(out_file)
        out_lblname = out_filename.replace('.bin', '.label')
        out_lbl = osp.join(output_lbl_dir, out_lblname)
        down_4_lbl_np.astype('uint32').tofile(out_lbl)

        
        down_pcd_xyz = down_4_pcd_np[:, :3]
        #print(down_pcd_xyz.shape)

        '''
        # convert numpy to open3d pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(down_pcd_xyz)

        
        out_dir = "down_test/"
        out_filename = velo_filename.split('/')[-1].replace(".bin", ".xyz")
        out_file = osp.join(out_dir, out_filename)
        o3d.io.write_point_cloud(out_file, pcd)

        #o3d.visualization.draw_geometries([pcd])
        
        '''


'''
velo_data_dir = "/media/4TB_HDD/Sem_Kitti/dataset/sequences/00/velodyne/"
filename_list = ["000000.bin", "000001.bin", "000002.bin", "000003.bin", "000004.bin", "000005.bin",
"000006.bin", "000007.bin", "000008.bin", "000009.bin", "000010.bin"]

file_list = []
label_list = []
for file_name in filename_list:
    file = osp.join(velo_data_dir, file_name)
    file_list.append(file)
    label = (file.replace('/velodyne/', '/labels/')).replace('.bin', '.label')
    label_list.append(label)
'''


"""
for velo_filename, velo_label in zip(file_list, label_list):
    # read binary data
    scan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
    label = (np.fromfile(velo_label, dtype=np.uint32)).reshape(-1, 1)
    print(scan.shape)
    #print(scan[0]) # [x, y, z, intensity]

    # get ring channel
    depth = np.linalg.norm(scan, 2, axis=1)
    pitch = np.arcsin(scan[:, 2] / depth) # arcsin(z, depth)
    #fov_down = -24.8 / 180.0 * np.pi
    fov_down = -25.0 / 180.0 * np.pi
    #fov = (abs(-24.8) + abs(2.0)) / 180.0 * np.pi
    fov = (abs(-25.0) + abs(3.0)) / 180.0 * np.pi
    proj_y = (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
    proj_y *= 64  # in [0.0, H]
    proj_y = np.floor(proj_y)
    proj_y = np.minimum(64 - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
    proj_y = proj_y.reshape(-1, 1)
    scan = np.concatenate((scan,proj_y), axis=1)
    scan = scan.tolist()
    for i in range(len(scan)):
        scan[i][-1] = int(scan[i][-1])

    #print(scan[0]) # [x, y, z, intensity, ring channel]


    down_4_pcd = []
    down_4_lbl = []
    for i in range(len(scan)):
        if scan[i][-1] % 4 == 0:
            down_4_pcd.append(scan[i])
            down_4_lbl.append(label[i])
            

    down_4_pcd_np = np.array(down_4_pcd)
    down_4_lbl_np = np.array(down_4_lbl)
    down_pcd_xyzr = down_4_pcd_np[:, :4].flatten()
    print(down_pcd_xyzr.shape)
    print(down_4_lbl_np.shape)

    out_dir = "down_test/"
    out_filename = velo_filename.split('/')[-1]
    out_file = osp.join(out_dir, out_filename)
    out_lbl = out_file.replace('.bin', '.label')
    # very very important to use .astype('float32')!!
    # or the result is very likely saved wrongly
    down_pcd_xyzr.astype('float32').tofile(out_file)
    down_4_lbl_np.astype('uint32').tofile(out_lbl)

    
    down_pcd_xyz = down_4_pcd_np[:, :3]
    #print(down_pcd_xyz.shape)

    '''
    # convert numpy to open3d pcd
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(down_pcd_xyz)

    
    out_dir = "down_test/"
    out_filename = velo_filename.split('/')[-1].replace(".bin", ".xyz")
    out_file = osp.join(out_dir, out_filename)
    o3d.io.write_point_cloud(out_file, pcd)

    #o3d.visualization.draw_geometries([pcd])
    
    '''

"""
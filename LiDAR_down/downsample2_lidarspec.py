# from https://github.com/guivenca/self-supervised-depth-completion
import numpy as np
import open3d as o3d

from glob import glob
from tqdm import tqdm
import os.path as osp
import os

def wrap_to_0_360(deg):
    while True:
        indices = np.nonzero(deg<0)[0]
        if len(indices)>0:
            deg[indices] = deg[indices] + 360
        else:
            break

    deg = ((100*deg).astype(int) % 36000) / 100.0
    return deg

def downsample_scan(velo, label, downsample_factor):
    """
    Downsample HDL-64 scans to the target_scan
    """
    x, y, z, r = velo[:,0], velo[:,1], velo[:,2], velo[:,3]
    dist_horizontal = (x**2 + y**2) ** 0.5
    # angles between the start of the scan (towards the rear)
    horizontal_degree = np.rad2deg(np.arctan2(y, x)) 
    horizontal_degree = wrap_to_0_360(horizontal_degree)

    scan_breakpoints = np.nonzero(np.diff(horizontal_degree) < -180)[0]+1
    scan_breakpoints = np.insert(scan_breakpoints, 0, 0)
    scan_breakpoints = np.append(scan_breakpoints, len(horizontal_degree)-1)
    num_scans = len(scan_breakpoints)-1

    # note that sometimes not all 64 scans show up in the image space
    indices = None
    if downsample_factor>1:
        indices = range(num_scans-downsample_factor//2, -1, -downsample_factor)
    else:
        indices = range(num_scans-1, -1, -1)

    assert num_scans <= 65, \
        "invalid number of scanlines: {}".format(num_scans)

    downsampled_velo = np.zeros(shape=[0, 4])
    downsampled_label = np.zeros(shape=[0, 1])
    for i in indices:
        start_index = scan_breakpoints[i]
        end_index = scan_breakpoints[i+1]
        scanline = velo[start_index:end_index, :]
        if label is not None:
            scanlbl = label[start_index:end_index, :]
        # the start of a scan is triggered at 180 degree
        #scan_time = wrap_to_0_360(horizontal_degree[start_index:end_index] + 180)/360
        #scanline = compensate_motion(scanline, scan_time, tx, ty, tz, roll, pitch, yaw)
        downsampled_velo = np.vstack((downsampled_velo, scanline))
        if label is not None:
            downsampled_label = np.vstack((downsampled_label, scanlbl))
    assert downsampled_velo.shape[0]>0, "downsampled velodyne has 0 measurements"
    if label is not None:
        return downsampled_velo, downsampled_label
    return downsampled_velo

def save_to_xyz(scan_pts, output_file_name):
    # convert numpy to open3d pcd
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan_pts)
    o3d.io.write_point_cloud(output_file_name, pcd)

'''
downsample0 = '/media/1TB_SSD/Sem_Kitti/down_dataset/sequences/00/velodyne/000000.bin'
velo_data_dir = '/media/1TB_SSD/Sem_Kitti/dataset/sequences/00/velodyne/'
velo_filename = velo_data_dir + '000000.bin'
label_file = (velo_filename.replace('/velodyne/', '/labels/')).replace('.bin', '.label')

scan0 = (np.fromfile(downsample0, dtype=np.float32)).reshape(-1, 4)
scan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
label = (np.fromfile(label_file, dtype=np.uint32)).reshape(-1, 1)
down_scan, down_lbl = downsample_scan(scan, label, downsample_factor=4)

print(scan0.shape)
print(scan.shape)
print(down_scan.shape)
print(down_lbl.shape)

down_scan.astype('float32').tofile('./scan_down.bin')
down_lbl.astype('uint32').tofile('./scan_down_lbl.label')
# convert numpy to open3d pcd
output_scan_xyz = './scan.xyz'
output_down_scan_xyz = './scan_down.xyz'
output0 = './scan0.xyz'
save_to_xyz(scan[:, :3], output_scan_xyz)
save_to_xyz(down_scan[:, :3], output_down_scan_xyz)
save_to_xyz(scan0[:, :3], output0)
'''
# no label in test seqs
test_list = ['11', '12', '13', '14',
            '15', '16', '17', '18', '19',
            '20', '21']
velo_data_dir = '/media/1TB_SSD/Sem_Kitti/dataset/sequences/'

#seq_list = ['00', '01', '02', '03', '04',
#            '05', '06', '07', '08', '09',
#            '10', '11', '12', '13', '14',
#            '15', '16', '17', '18', '19',
#            '20', '21']



down_factor = [2, 4]
output_data_dir = ['/media/1TB_SSD/Sem_Kitti/down_dataset_32/sequences', '/media/1TB_SSD/Sem_Kitti/down_dataset_16/sequences']

'''
seq_list = ['00', '01', '02', '03', '04',
            '05', '06', '07', '08', '09',
            '10']
# for train and val seqs (with labels)
file_list = []
label_list = []
for seq in seq_list:
    seq_files = sorted(glob(osp.join(velo_data_dir, seq, 'velodyne', '*.bin')))
    file_list.append(seq_files)
    seq_lbl = sorted(glob(osp.join(velo_data_dir, seq, 'labels', '*.label')))
    label_list.append(seq_lbl)


for i in range(len(down_factor)): 
    for j in range(len(file_list)):
        file_seq, label_seq = file_list[j], label_list[j]

        # setting output path
        seq = (file_seq[0]).split('/')[-3]
        output_file_dir = osp.join(output_data_dir[i], seq, 'velodyne')
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)
        output_lbl_dir = osp.join(output_data_dir[i], seq, 'labels')
        if not os.path.exists(output_lbl_dir):
            os.makedirs(output_lbl_dir)

        for k in tqdm(range(len((file_seq)))):
            velo_filename, velo_label = file_seq[k], label_seq[k]
            # read binary data
            scan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
            label = (np.fromfile(velo_label, dtype=np.uint32)).reshape(-1, 1)
            #print(scan.shape)
            #print(scan[0]) # [x, y, z, intensity]

            down_scan, down_lbl = downsample_scan(scan, label, downsample_factor=down_factor[i])
            #print(down_scan.shape)
            #print(down_lbl.shape)

            # very very important to use .astype('float32')!!
            # or the result is very likely saved wrongly
            out_filename = velo_filename.split('/')[-1]
            out_file = osp.join(output_file_dir, out_filename)
            down_scan.astype('float32').tofile(out_file)

            out_lblname = out_filename.replace('.bin', '.label')
            out_lbl = osp.join(output_lbl_dir, out_lblname)
            down_lbl.astype('uint32').tofile(out_lbl)
'''


# for test seqs (no label)
seq_list = ['11']
file_list = []
for seq in seq_list:
    seq_files = sorted(glob(osp.join(velo_data_dir, seq, 'velodyne', '*.bin')))
    file_list.append(seq_files)

for i in range(len(down_factor)): 
    for j in range(len(file_list)):
        file_seq = file_list[j]

        # setting output path
        seq = (file_seq[0]).split('/')[-3]
        output_file_dir = osp.join(output_data_dir[i], seq, 'velodyne')
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)

        for k in tqdm(range(len((file_seq)))):
            velo_filename = file_seq[k]
            # read binary data
            scan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)
            #print(scan.shape)
            #print(scan[0]) # [x, y, z, intensity]

            down_scan = downsample_scan(scan, None, downsample_factor=down_factor[i])
            #print(down_scan.shape)

            # very very important to use .astype('float32')!!
            # or the result is very likely saved wrongly
            out_filename = velo_filename.split('/')[-1]
            out_file = osp.join(output_file_dir, out_filename)
            down_scan.astype('float32').tofile(out_file)



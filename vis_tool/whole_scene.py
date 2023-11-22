import time
import argparse
from os.path import join, basename
from glob import glob

import yaml
from mayavi import mlab
import numpy as np

SEMANTIC_IDX2NAME = {
    1: 'unlabeled', 2: 'car', 3: 'bicycle', 4: 'motorcycle', 5: 'truck',
    6: 'other-vehicle', 7: 'person', 8: 'bicyclist', 9: 'motorcyclist', 
    10: 'road', 11: 'parking', 12: 'sidewalk', 13: 'other-ground', 
    14: 'building', 15: 'fence', 16: 'vegetation', 17: 'trunk',  
    18: 'terrain', 19: 'pole', 20: 'traffic-sign'
}

def process_lbl(annotated_data, lbl_map):
    with open(lbl_map, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    learning_map = semkittiyaml['learning_map'] # map to class 0-19
    thing_class = semkittiyaml['thing_class'] # instance class

    sem_data = annotated_data & 0xFFFF #delete high 16 digits binary(instance id)
    sem_data = np.vectorize(learning_map.__getitem__)(sem_data)
    inst_data = annotated_data
    #sem_cls = np.unique(sem_data, return_counts=False)
    #inst_type = np.unique(inst_data, return_counts=False)

    thing_list = []
    for cls, is_thing in thing_class.items():
        if is_thing:
            thing_list.append(cls)
    sem_is_thing = np.isin(sem_data, thing_list)
    #print(np.unique(sem_is_thing, return_counts=True))
    
    inst_data = inst_data.astype(np.int32)
    inst_data[np.where(sem_is_thing == False)] = -1
    unique_inst = np.unique(inst_data, return_counts=False)
    #print(np.unique(inst_data, return_counts=True))
    
    map_inst_dict = {} # key: new_id, value: old_id
    new_id = 0
    for old_id in unique_inst:
        if old_id == -1:
            continue
        inst_data[inst_data==old_id] = new_id
        map_inst_dict[new_id] = old_id
        new_id += 1
    
    #print(np.unique(inst_data, return_counts=True))
    return sem_data, inst_data, map_inst_dict

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

from typing import Tuple
from numba import njit
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
    # A.view(dtype) = [[(x1, y1, z1)],
    #                  [(x2, y2, z2)],
    #                   ...
    #                  [(xn, yn, zn)]]
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
# voxel miou utility
##################################################################################################################################
from chamferdist import ChamferDistance
import torch
# https://github.com/danielhavir/voxelize3d
@njit
def voxelize_jit_chf(
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
    points_copy = points_copy[mask, :]
    return points_copy

def chf_loss(upsampled_pts, gt_pts):
    '''
    upsampled_pts shape: [N, 3], np array
    gt_pts shape: [N, 3], np array
    '''
    chamferDist = ChamferDistance()
    upsampled_pts = torch.from_numpy(upsampled_pts).type(torch.FloatTensor).cuda().unsqueeze(0)
    gt_pts = torch.from_numpy(gt_pts).type(torch.FloatTensor).cuda().unsqueeze(0)
    #print("gt_pts.shape: ", gt_pts.shape)
    #print("upsampled_pts.shape: ", upsampled_pts.shape)

    forward = chamferDist(upsampled_pts, gt_pts, reduction=None)
    backward = chamferDist(gt_pts, upsampled_pts, reduction=None)

    #print('forward: ', forward)
    #print('backward: ', backward)
    chamfer_loss = forward.mean() + backward.mean()

    return chamfer_loss

# Color lut(lookup table)
# Integer index in s corresponding to the point 
# will be used as the row in the lookup table
lut = np.zeros((2, 4))
lut[0, :] = [255, 255, 255, 255] # white
lut[1, :] = [255, 0, 0, 255] # red

# filter condition
ins_out_range = 0.3
voxel_size = np.array([0.2, 0.2, 0.2])

class Multi_Vis:
    def __init__(self, input_list, point_list1, point_list2, point_list3, 
                 src_list, lbl_list, lbl_map,
                 request_cls='car', bg_clr=None, scene_size=None, ins_size=None):
        self._id = 0
        self.inst_id = 0
        self.in_list = input_list
        self.point_list1 = point_list1
        self.point_list2 = point_list2
        self.point_list3 = point_list3
        self.src_list = src_list
        self.lbl_list = lbl_list
        self.lbl_map = lbl_map
        self.points = None # np array of current point cloud
        self.request_cls = request_cls

        if bg_clr is not None:
            back_clr = bg_clr
        else:
            back_clr = (0, 0, 0) # black
        if scene_size is not None:
            sce_size = scene_size
        else:
            sce_size = (600, 500)
        if ins_size is not None:
            inst_size = ins_size
        else:
            inst_size = (600, 500)

        # Create figure for features
        self.fig1 = mlab.figure('Grond Truth', bgcolor=back_clr, size=sce_size)
        self.fig1.scene.parallel_projection = False
        self.fig2 = mlab.figure('Model1', bgcolor=back_clr, size=sce_size)
        self.fig2.scene.parallel_projection = False
        self.fig3 = mlab.figure('Model2', bgcolor=back_clr, size=sce_size)
        self.fig3.scene.parallel_projection = False
        self.fig7 = mlab.figure('Input', bgcolor=back_clr, size=sce_size)
        self.fig7.scene.parallel_projection = False
        
        self.fig4 = mlab.figure('GT Instance', bgcolor=back_clr, size=inst_size)
        self.fig4.scene.parallel_projection = False
        self.fig5 = mlab.figure('Instance1', bgcolor=back_clr, size=inst_size)
        self.fig5.scene.parallel_projection = False
        self.fig6 = mlab.figure('Instance2', bgcolor=back_clr, size=inst_size)
        self.fig6.scene.parallel_projection = False
        self.fig8 = mlab.figure('Input Instance', bgcolor=back_clr, size=inst_size)
        self.fig8.scene.parallel_projection = False

        self.update_scene()
        self.mean_inst_xyz, self.inst_class = self.find_ins_xyz()

    def update_scene(self):
        #  clear figure
        mlab.clf(self.fig1)
        mlab.clf(self.fig2)
        mlab.clf(self.fig3)
        mlab.clf(self.fig7)
        # Load points
        points1_name = self.point_list1[self._id]
        #print(points1_name)
        if basename(points1_name).endswith('.bin'):
            self.points1 = np.fromfile(points1_name, dtype=np.float32).reshape(-1, 4)
        elif basename(points1_name).endswith('.xyz'):
            self.points1 = np.loadtxt(points1_name)
        self.points1 = self.points1[:, :3]
        points2_name = self.point_list2[self._id]
        if basename(points2_name).endswith('.bin'):
            self.points2 = np.fromfile(points2_name, dtype=np.float32).reshape(-1, 4)
        elif basename(points2_name).endswith('.xyz'):
            self.points2 = np.loadtxt(points2_name)
        self.points2 = self.points2[:, :3]
        points3_name = self.point_list3[self._id]
        if basename(points3_name).endswith('.bin'):
            self.points3 = np.fromfile(points3_name, dtype=np.float32).reshape(-1, 4)
        elif basename(points3_name).endswith('.xyz'):
            self.points3 = np.loadtxt(points3_name)
        self.points3 = self.points3[:, :3]
        pointsin_name = self.in_list[self._id]
        if basename(pointsin_name).endswith('.bin'):
            self.pointsin = np.fromfile(pointsin_name, dtype=np.float32).reshape(-1, 4)
        elif basename(pointsin_name).endswith('.xyz'):
            self.pointsin = np.loadtxt(pointsin_name)
        self.pointsin = self.pointsin[:, :3]


        # Show point clouds colorized with activations
        color_1 = np.copy(self.points1[:, 2])
        color_1[color_1 < -3] = -3
        color_1[color_1 > 1] = 1
        self.activations1 = mlab.points3d(self.points1[:, 0],
                                          self.points1[:, 1],
                                          self.points1[:, 2],
                                          #np.zeros((self.points1.shape[0])),
                                          #self.points1[:, 2],
                                          color_1,
                                          colormap="jet",
                                          mode="point",
                                          scale_factor=3.0,
                                          scale_mode='none',
                                          figure=self.fig1)
        self.activations1.actor.property.render_points_as_spheres =  True
        self.activations1.actor.property.point_size = 2.5
        #self.activations1.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut)
        #self.activations1.module_manager.scalar_lut_manager.lut.table = lut
        # New title
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.title(str(self._id), color=(0, 0, 0), size=0.3, height=0.01)
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        color_2 = np.copy(self.points2[:, 2])
        color_2[color_2 < -3] = -3
        color_2[color_2 > 1] = 1
        self.activations2 = mlab.points3d(self.points2[:, 0],
                                          self.points2[:, 1],
                                          self.points2[:, 2],
                                          #np.zeros((self.points2.shape[0])),
                                          color_2,
                                          colormap="jet",
                                          mode="point",
                                          scale_factor=3.0,
                                          scale_mode='none',
                                          figure=self.fig2)
        self.activations2.actor.property.render_points_as_spheres =  True
        self.activations2.actor.property.point_size = 2.5
        #self.activations2.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut)
        #self.activations2.module_manager.scalar_lut_manager.lut.table = lut
        # New title
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.title(str(self._id), color=(0, 0, 0), size=0.3, height=0.01)
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        color_3 = np.copy(self.points3[:, 2])
        color_3[color_3 < -3] = -3
        color_3[color_3 > 1] = 1
        self.activations3 = mlab.points3d(self.points3[:, 0],
                                          self.points3[:, 1],
                                          self.points3[:, 2],
                                          #np.zeros((self.points3.shape[0])),
                                          color_3,
                                          colormap="jet",
                                          mode="point",
                                          scale_factor=3.0,
                                          scale_mode='none',
                                          figure=self.fig3)
        self.activations3.actor.property.render_points_as_spheres =  True
        self.activations3.actor.property.point_size = 2.5
        #self.activations3.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut)
        #self.activations3.module_manager.scalar_lut_manager.lut.table = lut
        # New title
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.title(str(self._id), color=(0, 0, 0), size=0.3, height=0.01)
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        color_in = np.copy(self.pointsin[:, 2])
        color_in[color_in < -3] = -3
        color_in[color_in > 1] = 1
        self.activations7 = mlab.points3d(self.pointsin[:, 0],
                                          self.pointsin[:, 1],
                                          self.pointsin[:, 2],
                                          #np.zeros((self.pointsin.shape[0])),
                                          color_in,
                                          colormap="jet",
                                          mode="point",
                                          scale_factor=3.0,
                                          scale_mode='none',
                                          figure=self.fig7)
        self.activations7.actor.property.render_points_as_spheres =  True
        self.activations7.actor.property.point_size = 2
        #self.activations7.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut)
        #self.activations7.module_manager.scalar_lut_manager.lut.table = lut
        # New title
        text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        mlab.title(str(self._id), color=(0, 0, 0), size=0.3, height=0.01)
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        mlab.orientation_axes()

        return
    
    def update_ins(self, ins_pos):
        #  clear figure
        mlab.clf(self.fig4)
        mlab.clf(self.fig5)
        mlab.clf(self.fig6)
        mlab.clf(self.fig8)
        if self.request_cls == 'car':
            radius = 5
        elif self.request_cls == 'person':
            radius = 1.5
        elif self.request_cls == 'bicyclist':
            radius = 2

        ins1_mask = filter_radius(self.points1, radius=radius, center=ins_pos)
        ins2_mask = filter_radius(self.points2, radius=radius, center=ins_pos)
        ins3_mask = filter_radius(self.points3, radius=radius, center=ins_pos)
        ins4_mask = filter_radius(self.pointsin, radius=radius, center=ins_pos)
        self.ins_pts1 = self.points1[ins1_mask]
        #print(self.ins_pts1.shape)
        self.ins_pts2 = self.points2[ins2_mask]
        #print(self.ins_pts2.shape)
        self.ins_pts3 = self.points3[ins3_mask]
        self.ins_pts4 = self.pointsin[ins4_mask]
        #print(self.ins_pts4.shape)

        max_xyz = np.max(self.ins_pts1, axis=0)
        min_xyz = np.min(self.ins_pts1, axis=0)
        # fit xyz to the voxel grid size(0.5)
        max_xyz = (np.ceil(max_xyz/0.2)*0.2+ins_out_range).reshape(-1)
        min_xyz = (np.floor(min_xyz/0.2)*0.2-ins_out_range).reshape(-1)
        min_xyz[2] = min_xyz[2] + ins_out_range # remove floor
        scene_range = np.concatenate((min_xyz, max_xyz))
        
        ## cal IoU
        coor_gt = voxelize_jit(self.ins_pts1, voxel_size, scene_range)
        unique_voxel_gt = np.unique(coor_gt, axis=0)
        #print(len(unique_voxel_ins))
        coor_ins = voxelize_jit(self.ins_pts2, voxel_size, scene_range)
        unique_voxel_ins = np.unique(coor_ins, axis=0)
        coor_p = voxelize_jit(self.ins_pts3, voxel_size, scene_range)
        unique_voxel_p = np.unique(coor_p, axis=0)
        ins_miou_i = miou(unique_voxel_ins, unique_voxel_gt)
        p_miou_i = miou(unique_voxel_p, unique_voxel_gt)

        # cal_chf
        coor_gt2 = voxelize_jit_chf(self.ins_pts1, voxel_size, scene_range)
        unique_voxel_gt2 = np.unique(coor_gt2, axis=0)
        #print(len(unique_voxel_ins))
        coor_ins2 = voxelize_jit_chf(self.ins_pts2, voxel_size, scene_range)
        unique_voxel_ins2 = np.unique(coor_ins2, axis=0)
        coor_p2 = voxelize_jit_chf(self.ins_pts3, voxel_size, scene_range)
        unique_voxel_p2 = np.unique(coor_p2, axis=0)
        ins_chf_i = chf_loss(unique_voxel_ins2, unique_voxel_gt2).item()
        #print(f'ins_chf_i: {ins_chf_i:.4f}')
        p_chf_i = chf_loss(unique_voxel_p2, unique_voxel_gt2).item()
        #print(f'p_chf_i: {p_chf_i}')

        # Show point clouds colorized with activations
        color_4 = np.copy(self.ins_pts1[:, 2])
        color_4[color_4 < -3] = -3
        color_4[color_4 > 1] = 1
        self.activations4 = mlab.points3d(self.ins_pts1[:, 0],
                                          self.ins_pts1[:, 1],
                                          self.ins_pts1[:, 2],
                                          #np.zeros((self.ins_pts1.shape[0])),
                                          color_4,
                                          mode="point",
                                          scale_factor=3.0,
                                          scale_mode='none',
                                          figure=self.fig4)
        self.activations4.actor.property.render_points_as_spheres = True
        self.activations4.actor.property.point_size = 3
        #self.activations4.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut)
        #self.activations4.module_manager.scalar_lut_manager.lut.table = lut

        color_5 = np.copy(self.ins_pts2[:, 2])
        color_5[color_5 < -3] = -3
        color_5[color_5 > 1] = 1
        self.activations5 = mlab.points3d(self.ins_pts2[:, 0],
                                          self.ins_pts2[:, 1],
                                          self.ins_pts2[:, 2],
                                          #np.zeros((self.ins_pts2.shape[0])),
                                          color_5,
                                          mode="point",
                                          scale_factor=3.0,
                                          scale_mode='none',
                                          figure=self.fig5)
        self.activations5.actor.property.render_points_as_spheres = True
        self.activations5.actor.property.point_size = 3
        #self.activations5.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut)
        #self.activations5.module_manager.scalar_lut_manager.lut.table = lut
        # Add title
        text = '<--- (press z for previous)' + 70 * ' ' + '(press x for next) --->'
        mlab.title(f'IoU: {ins_miou_i:.4f} / Chf: {ins_chf_i:.4f}', color=(0, 0, 0), size=0.3, height=0.01, figure=self.fig5)
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98, figure=self.fig5)
        mlab.orientation_axes(figure=self.fig5)

        color_6 = np.copy(self.ins_pts3[:, 2])
        color_6[color_6 < -3] = -3
        color_6[color_6 > 1] = 1
        self.activations6 = mlab.points3d(self.ins_pts3[:, 0],
                                          self.ins_pts3[:, 1],
                                          self.ins_pts3[:, 2],
                                          #np.zeros((self.ins_pts3.shape[0])),
                                          color_6,
                                          mode="point",
                                          scale_factor=3.0,
                                          scale_mode='none',
                                          figure=self.fig6)
        self.activations6.actor.property.render_points_as_spheres =  True
        self.activations6.actor.property.point_size = 3
        #self.activations6.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut)
        #self.activations6.module_manager.scalar_lut_manager.lut.table = lut
        # Add title
        text = '<--- (press z for previous)' + 70 * ' ' + '(press x for next) --->'
        mlab.title(f'IoU: {p_miou_i:.4f} / Chf: {p_chf_i:.4f}', color=(0, 0, 0), size=0.3, height=0.01, figure=self.fig6)
        mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98, figure=self.fig6)
        mlab.orientation_axes(figure=self.fig6)

        # Show point clouds colorized with activations
        color_8 = np.copy(self.ins_pts4[:, 2])
        color_8[color_8 < -3] = -3
        color_8[color_8 > 1] = 1
        self.activations8 = mlab.points3d(self.ins_pts4[:, 0],
                                          self.ins_pts4[:, 1],
                                          self.ins_pts4[:, 2],
                                          #np.zeros((self.ins_pts4.shape[0])),
                                          color_8,
                                          mode="point",
                                          scale_factor=3.0,
                                          scale_mode='none',
                                          figure=self.fig8)
        self.activations8.actor.property.render_points_as_spheres =  True
        self.activations8.actor.property.point_size = 3
        #self.activations8.module_manager.scalar_lut_manager.lut.number_of_colors = len(lut)
        #self.activations8.module_manager.scalar_lut_manager.lut.table = lut
        return
    
    def keyboard_callback(self, vtk_obj, event):
        '''
        KeyEvent:
        - G: previous frame
        - H: next frame
        - Z: previous instance view
        - X: next instance view
        '''
        if vtk_obj.GetKeyCode() in ['g', 'G']:
            self._id = (self._id - 1) % len(self.point_list1)
            self.update_scene()
            self.mean_inst_xyz, self.inst_class = self.find_ins_xyz()
        elif vtk_obj.GetKeyCode() in ['h', 'H']:
            self._id = (self._id + 1) % len(self.point_list1)
            self.update_scene()
            self.mean_inst_xyz, self.inst_class = self.find_ins_xyz()

        if vtk_obj.GetKeyCode() in ['z', 'Z']:
            self.inst_id = (self.inst_id - 1) % len(self.mean_inst_xyz)
            ins_mean_pos = self.mean_inst_xyz[self.inst_id]
            ins_cls = self.inst_class[self.inst_id]
            print(f'Current Class is {ins_cls}')
            ## add red center
            self.update_ins(ins_mean_pos)
            #self.add_center(ins_mean_pos)
            mlab.view(azimuth=180,
                      elevation=70,
                      distance=40.0,
                      focalpoint=ins_mean_pos,
                      figure=self.fig1)
            mlab.view(azimuth=180,
                      elevation=70,
                      distance=20.0,
                      focalpoint=ins_mean_pos,
                      figure=self.fig3)
            
        elif vtk_obj.GetKeyCode() in ['x', 'X']:
            self.inst_id = (self.inst_id + 1) % len(self.mean_inst_xyz)
            ins_mean_pos = self.mean_inst_xyz[self.inst_id]
            ins_cls = self.inst_class[self.inst_id]
            print(f'Current Class is {ins_cls}')
            ## add red center
            self.update_ins(ins_mean_pos)
            #self.add_center(ins_mean_pos)
            mlab.view(azimuth=180,
                      elevation=70,
                      distance=40.0,
                      focalpoint=ins_mean_pos,
                      figure=self.fig1)
            mlab.view(azimuth=180,
                      elevation=70,
                      distance=20.0,
                      focalpoint=ins_mean_pos,
                      figure=self.fig3)
        return
    
    def find_ins_xyz(self, min_pts=20):
        '''
        Find the instance positoin and class from the label
        '''
        src_name = self.src_list[self._id]
        #print(src_name)
        self.src_points = np.fromfile(src_name, dtype=np.float32).reshape(-1, 4)
        self.src_points = self.src_points[:, :3]
        lbl_name = self.lbl_list[self._id]
        #print(lbl_name)
        lbl = np.fromfile(lbl_name, dtype=np.uint32).reshape(-1)
        sem_lbl, inst_lbl, map_inst_dict = process_lbl(lbl, self.lbl_map)

        ## find label xyz and name
        max_inst_xyz = []
        #mean_inst_xyz = []
        inst_class = []
        unique_inst = np.unique(inst_lbl, return_counts=False)
        #print(np.unique(inst_lbl, return_counts=True))
        for inst_id in unique_inst:
            if inst_id == -1: # background
                continue
            inst_xyzs = self.src_points[inst_lbl == inst_id]
            #print(sem_lbl[inst_lbl == inst_id])
            inst_cls = SEMANTIC_IDX2NAME[sem_lbl[inst_lbl == inst_id][0]+1]
            if inst_xyzs.shape[0] < 20 or inst_cls != self.request_cls:
                continue
            max_xyz = np.max(inst_xyzs, axis=0)[:3]
            #mean_xyz = np.mean(inst_xyzs, axis=0)[:3]
            max_inst_xyz.append(max_xyz)
            #mean_inst_xyz.append(mean_xyz)
            inst_class.append(inst_cls)
        print('Finishing finding the instances..')
        #print(f'Total {len(mean_inst_xyz)} in the scene..')
        print(f'Total {len(max_inst_xyz)} in the scene..')
        #return mean_inst_xyz, inst_class
        return max_inst_xyz, inst_class
    
    def add_center(self, pos):
        ## add red center
        new_point_data1 = np.vstack((self.points1, pos))
        new_point_data2 = np.vstack((self.points2, pos))
        new_point_data3 = np.vstack((self.points3, pos))
        new_point_data4 = np.vstack((self.ins_pts1, pos))
        new_point_data5 = np.vstack((self.ins_pts2, pos))
        new_point_data6 = np.vstack((self.ins_pts3, pos))
        new_color1 = np.concatenate((np.zeros((self.points1.shape[0])), np.array([1])))
        new_color2 = np.concatenate((np.zeros((self.points2.shape[0])), np.array([1])))
        new_color3 = np.concatenate((np.zeros((self.points3.shape[0])), np.array([1])))
        new_color4 = np.concatenate((np.zeros((self.ins_pts1.shape[0])), np.array([1])))
        new_color5 = np.concatenate((np.zeros((self.ins_pts2.shape[0])), np.array([1])))
        new_color6 = np.concatenate((np.zeros((self.ins_pts3.shape[0])), np.array([1])))
        self.activations1.mlab_source.set(points=new_point_data1)
        self.activations1.mlab_source.scalars = new_color1
        self.activations2.mlab_source.set(points=new_point_data2)
        self.activations2.mlab_source.scalars = new_color2
        self.activations3.mlab_source.set(points=new_point_data3)
        self.activations3.mlab_source.scalars = new_color3
        self.activations4.mlab_source.set(points=new_point_data4)
        self.activations4.mlab_source.scalars = new_color4
        self.activations5.mlab_source.set(points=new_point_data5)
        self.activations5.mlab_source.scalars = new_color5
        self.activations6.mlab_source.set(points=new_point_data6)
        self.activations6.mlab_source.scalars = new_color6
    
    def run(self):
        self.fig1.scene.interactor.add_observer('KeyPressEvent', self.keyboard_callback)
        self.fig2.scene.interactor.add_observer('KeyPressEvent', self.keyboard_callback)
        self.fig3.scene.interactor.add_observer('KeyPressEvent', self.keyboard_callback)
        self.fig4.scene.interactor.add_observer('KeyPressEvent', self.keyboard_callback)
        self.fig5.scene.interactor.add_observer('KeyPressEvent', self.keyboard_callback)
        self.fig6.scene.interactor.add_observer('KeyPressEvent', self.keyboard_callback)
        self.fig7.scene.interactor.add_observer('KeyPressEvent', self.keyboard_callback)
        self.fig8.scene.interactor.add_observer('KeyPressEvent', self.keyboard_callback)
        
        
        mlab.sync_camera(self.fig1, self.fig2)
        mlab.sync_camera(self.fig2, self.fig1)
        mlab.sync_camera(self.fig1, self.fig3)
        mlab.sync_camera(self.fig3, self.fig1)
        mlab.sync_camera(self.fig1, self.fig7)
        mlab.sync_camera(self.fig7, self.fig1)

        mlab.sync_camera(self.fig1, self.fig4)
        mlab.sync_camera(self.fig4, self.fig1)

        mlab.sync_camera(self.fig4, self.fig5)
        mlab.sync_camera(self.fig5, self.fig4)
        mlab.sync_camera(self.fig4, self.fig6)
        mlab.sync_camera(self.fig6, self.fig4)
        mlab.sync_camera(self.fig4, self.fig8)
        mlab.sync_camera(self.fig8, self.fig4)
        
        mlab.show()

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, default='/media/1TB_SSD/Sem_Kitti/dataset/sequences/',
                        help='specify the gt(64-beam) point cloud data file or directory')
    parser.add_argument('--input_path', type=str, default='/media/1TB_SSD/Sem_Kitti/down_dataset_32/sequences/',
                        help='specify the input(32-beam) point cloud data file or directory')
    parser.add_argument('--lbl_map', type=str, default='/media/1TB_SSD/Sem_Kitti/semantic-kitti.yaml', 
                        help='Semkitti class lbl mapping')
    parser.add_argument('--sr_path', type=str, 
                        default='/media/1TB_SSD/all_seq_result/',
                        help='specify the SR point cloud data file or directory')
    parser.add_argument('--model', type=str, 
                        default='Ins_aux_result',
                        help='The model result, choice: [Ins_aux_result, mpu, npt_result, pugcn]')
    parser.add_argument('--model2', type=str, 
                        default='npt_result',
                        help='The model result, choice: [Ins_aux_result, mpu, npt_result, pugcn]. None is set input')
    parser.add_argument('--seq', type=int, default=8, help='Sequence used in Semkitti.')
    parser.add_argument('--cls', type=str, default='car', help='Require class for instance [car, person, bicyclist]')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_config()
    lbl_map = args.lbl_map
    root_path = args.data_path
    in_path = args.input_path

    #pcd_list = sorted(glob(join(root_path, str(args.seq).zfill(2), 'velodyne', '*.bin')))
    #lbl_list = sorted(glob(join(root_path, str(args.seq).zfill(2), 'labels', '*.label')))
    sr_list = sorted(glob(join(args.sr_path, str(args.seq).zfill(2), args.model,'*.xyz')))
    #sr_list = sorted(glob(join('/media/4TB_HDD/Ins_aux_result/no_stuff_epoch_10_ckpt/','*.xyz')))
    pcd_list = [join(root_path, str(args.seq).zfill(2), 'velodyne', basename(i).replace('.xyz', '.bin')) 
                for i in sr_list]
    input_list = [join(in_path, str(args.seq).zfill(2), 'velodyne', basename(i).replace('.xyz', '.bin')) 
                for i in sr_list]
    lbl_list = [join(root_path, str(args.seq).zfill(2), 'labels', basename(i).replace('.xyz', '.label')) 
                for i in sr_list]
    
    if args.model2 is not None:
        sr_list2 = sorted(glob(join(args.sr_path, str(args.seq).zfill(2), args.model2,'*.xyz')))
    else:
        sr_list2 = pcd_list
    
    #print(pcd_list)
    #print(lbl_list)

    app = Multi_Vis(input_list, pcd_list, sr_list, sr_list2, pcd_list, lbl_list, lbl_map, args.cls
                    , bg_clr=(1, 1, 1), scene_size=(1024, 768), ins_size=None)
    #app = Multi_Vis(pcd_list, sr_list, sr_list, pcd_list, lbl_list, lbl_map, 'person')
    app.run()
    

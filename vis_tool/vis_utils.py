# vis two point cloud
from os.path import join, basename
from glob import glob

from mayavi import mlab
import numpy as np


# Color lut(lookup table)
# Integer index in s corresponding to the point 
# will be used as the row in the lookup table
lut = np.zeros((2, 4))
lut[0, :] = [0, 0, 0, 122] # black
lut[1, :] = [255, 0, 0, 255] # red

class Multi_Vis:
    def __init__(self, point_list1, point_list2, point_list3=None,
                 bg_clr=None, scene_size=None, mode=0):
        self._id = 0
        self.point_list1 = point_list1
        self.point_list2 = point_list2
        self.point_list3 = point_list3
        self.mode = mode

        if bg_clr is not None:
            back_clr = bg_clr
        else:
            back_clr = (0, 0, 0) # black
        if scene_size is not None:
            sce_size = scene_size
        else:
            sce_size = (600, 500)

        

        if self.mode == 1:
            self.fig1 = mlab.figure('NOUP Scene', bgcolor=back_clr, size=sce_size)
            self.fig1.scene.parallel_projection = False
            self.fig2 = mlab.figure('GT Points', bgcolor=back_clr, size=sce_size)
            self.fig2.scene.parallel_projection = False
            self.fig3 = mlab.figure('UP Points', bgcolor=back_clr, size=sce_size)
            self.fig3.scene.parallel_projection = False
        else:
            # Create figure for features
            self.fig1 = mlab.figure('Pred Scene', bgcolor=back_clr, size=sce_size)
            self.fig1.scene.parallel_projection = False
            self.fig2 = mlab.figure('GT Points', bgcolor=back_clr, size=sce_size)
            self.fig2.scene.parallel_projection = False

        self.update_scene()

    def update_scene(self, point_size=5.0):
        #  clear figure
        mlab.clf(self.fig1)
        mlab.clf(self.fig2)
        if self.mode == 1:
            mlab.clf(self.fig3)
        # Load points
        points1_name = self.point_list1[self._id]
        #print(points1_name)
        if basename(points1_name).endswith('.bin'):
            self.points1 = np.fromfile(points1_name, dtype=np.float32).reshape(-1, 4)
        elif basename(points1_name).endswith('.xyz'):
            self.points1 = np.loadtxt(points1_name)
        #self.points1 = self.points1[:, :3]

        points2_name = self.point_list2[self._id]
        if basename(points2_name).endswith('.bin'):
            self.points2 = np.fromfile(points2_name, dtype=np.float32).reshape(-1, 4)
        elif basename(points2_name).endswith('.xyz'):
            self.points2 = np.loadtxt(points2_name)
        #self.points2 = self.points2[:, :3]

        if self.mode == 1:
            points3_name = self.point_list3[self._id]
            if basename(points3_name).endswith('.bin'):
                self.points3 = np.fromfile(points3_name, dtype=np.float32).reshape(-1, 4)
            elif basename(points3_name).endswith('.xyz'):
                self.points3 = np.loadtxt(points3_name)
            #self.points2 = self.points2[:, :3]


        total_cls = np.unique(self.points1[:, 3])
        #print(np.unique(self.points1[:, 3]))
        cnt = 0
        for cls_id in total_cls:
            mask = np.where(self.points1[:, 3] == cls_id)[0]
            #print(mask.shape)
            self.points1[mask, 3] = cnt
            cnt += 1
        #print(np.unique(self.points1[:, 3]))

        total_cls = np.unique(self.points2[:, 3])
        #print(np.unique(self.points2[:, 3]))
        cnt = 0
        for cls_id in total_cls:
            mask = np.where(self.points2[:, 3] == cls_id)[0]
            #print(mask.shape)
            self.points2[mask, 3] = cnt
            cnt += 1
        #print(np.unique(self.points2[:, 3]))

        if self.mode == 1:
            total_cls = np.unique(self.points3[:, 3])
            #print(np.unique(self.points3[:, 3]))
            cnt = 0
            for cls_id in total_cls:
                mask = np.where(self.points3[:, 3] == cls_id)[0]
                #print(mask.shape)
                self.points3[mask, 3] = cnt
                cnt += 1
            #print(np.unique(self.points3[:, 3]))

        # Show point clouds colorized with activations
        color_1 = np.copy(self.points1[:, 2])
        color_1[color_1 < -3] = -3
        color_1[color_1 > 1] = 1
        self.activations1 = mlab.points3d(self.points1[:, 0],
                                          self.points1[:, 1],
                                          self.points1[:, 2],
                                          #np.zeros((self.points1.shape[0])),
                                          self.points1[:, 3],
                                          #color_1,
                                          colormap="jet",
                                          mode="point",
                                          scale_factor=3.0,
                                          scale_mode='none',
                                          figure=self.fig1)
        self.activations1.actor.property.render_points_as_spheres = True
        self.activations1.actor.property.point_size = point_size
        # New title
        #text = '<--- (press g for previous)' + 50 * ' ' + '(press h for next) --->'
        #mlab.title(str(self._id), color=(0, 0, 0), size=0.3, height=0.01)
        #mlab.text(0.01, 0.01, text, color=(0, 0, 0), width=0.98)
        #mlab.orientation_axes()

        color_2 = np.copy(self.points2[:, 2])
        color_2[color_2 < -3] = -3
        color_2[color_2 > 1] = 1
        self.activations2 = mlab.points3d(self.points2[:, 0],
                                          self.points2[:, 1],
                                          self.points2[:, 2],
                                          #np.zeros((self.points2.shape[0])),
                                          self.points2[:, 3],
                                          #color_2,
                                          colormap="jet",
                                          mode="point",
                                          scale_factor=3.0,
                                          scale_mode='none',
                                          figure=self.fig2)
        self.activations2.actor.property.render_points_as_spheres = True
        self.activations2.actor.property.point_size = point_size
        mlab.orientation_axes()

        if self.mode == 1:
            color_3 = np.copy(self.points3[:, 2])
            color_3[color_3 < -3] = -3
            color_3[color_3 > 1] = 1
            self.activations3 = mlab.points3d(self.points3[:, 0],
                                            self.points3[:, 1],
                                            self.points3[:, 2],
                                            #np.zeros((self.points2.shape[0])),
                                            self.points3[:, 3],
                                            #color_2,
                                            colormap="jet",
                                            mode="point",
                                            scale_factor=3.0,
                                            scale_mode='none',
                                            figure=self.fig3)
            self.activations3.actor.property.render_points_as_spheres = True
            self.activations3.actor.property.point_size = point_size


        # mouse click to get position
        self.glyph_points = self.activations2.glyph.glyph_source.glyph_source.output.points.to_array()
        # register the mouse click action
        picker = self.fig2.on_mouse_pick(self.picker_callback)
    
    def keyboard_callback(self, vtk_obj, event):
        '''
        KeyEvent:
        - G: previous frame
        - H: next frame
        '''
        if vtk_obj.GetKeyCode() in ['g', 'G']:
            self._id = (self._id - 1) % len(self.point_list1)
            self.update_scene()
        elif vtk_obj.GetKeyCode() in ['h', 'H']:
            self._id = (self._id + 1) % len(self.point_list1)
            self.update_scene()
        return
    
    def picker_callback(self, picker):
        # https://stackoverflow.com/questions/41716032/retrieve-data-on-mouse-click-in-mayavi
        # mouse click to get the input coordinate
        if picker.actor in self.activations7.actor.actors:
            point_id = picker.point_id//self.glyph_points.shape[0]
            print("{}:".format(point_id))
            if point_id != -1:
                #print("{}:".format(point_id))
                print("({} {} {}) ".format(
                    self.pointsin[point_id, 0], self.pointsin[point_id, 1], self.pointsin[point_id, 2]))
    
    def run(self):
        self.fig1.scene.interactor.add_observer('KeyPressEvent', self.keyboard_callback)
        self.fig2.scene.interactor.add_observer('KeyPressEvent', self.keyboard_callback)
        if self.mode == 1:
            self.fig2.scene.interactor.add_observer('KeyPressEvent', self.keyboard_callback)
        # sync scene view
        mlab.sync_camera(self.fig1, self.fig2)
        mlab.sync_camera(self.fig2, self.fig1)
        if self.mode == 1:
            mlab.sync_camera(self.fig3, self.fig1)
            mlab.sync_camera(self.fig1, self.fig3)
        mlab.show()


if __name__ == '__main__':
    #dir = '/media/1TB_HDD/ins_seg_test/testing_scene/'
    
    dir = '/media/1TB_HDD/ins_seg_test/noup_ins_div/near'
    dir2 = '/media/1TB_HDD/ins_seg_test/up2_ins_div/near'
    #dir = '/media/1TB_HDD/ins_seg_test/noup_ins_div/mid'
    #dir2 = '/media/1TB_HDD/ins_seg_test/up_ins_div/mid'
    #dir = '/media/1TB_HDD/ins_seg_test/noup_ins_div/far'
    #dir2 = '/media/1TB_HDD/ins_seg_test/up_ins_div/far'


    pred_list = sorted(glob(join(dir,'*_ins.xyz')))
    gt_list = sorted(glob(join(dir,'*_insgt.xyz')))
    pred_list2 = sorted(glob(join(dir2,'*_ins.xyz')))

    app = Multi_Vis(pred_list, gt_list, pred_list2, bg_clr=(1, 1, 1), scene_size=(450, 450), mode=1)
    app.run()


# INS_ATT
- [paper](https://ieeexplore.ieee.org/abstract/document/10801358)
## Installation
I borrow codes from [NPT](https://github.com/WanquanF/NeuralPoints), [chamferdist](https://github.com/krrish94/chamferdist), [instance-segmentation-pytorch](https://github.com/Wizaron/instance-segmentation-pytorch) and [Pointformer](https://github.com/Vladimir2506/Pointformer).
1. I assume that you have installed pytorch with corresponding torchvision and cudatoolkit.
2. Compile pointnet operation.
~~~
cd [ProjectPath]/pointnet2
python setup.py install
~~~
3. Compile the chamfer distance operation.
~~~
cd chamferdist/
python setup.py install
~~~
Note that chamfer distance calculation in chamferdist/chamferdist/chamfer.py should be modified if you followed the formulas like this.
![image](https://github.com/willy8898/INS_ATT/assets/62001022/1f5564f7-a3a8-4c0a-8c08-a1c3424c248d)

## Dataset preparation
1. Downsample LiDAR scanning
~~~
# downsample LiDAR scene
cd LiDAR_down
python downsample2_lidarspec.py --data_path [Semantic-KITTI folder] --save_path [downsample dataset folder]
~~~
2. Split to patches
~~~
# prepare the training set
cd dataset_prepare
python scene_preprocess.py --dense_path [Semantic-KITTI folder] --sparse_path [downsample dataset folder] \
--save_path [where you want to save your patches training set]
~~~

## Training
~~~
python train.py --data_path [where your processed dataset is]
~~~

## Inference
~~~
cd inference
python infer_kitti_ins_allseq --input_path [downsample LiDAR scene path] --seq [Semantic-KITTI sequence] \
--ckpt_path [checkpoint path] --save_path [where to save predict scene] \
--frame_path [optional]
~~~

## Evaluation
~~~
cd eval
# chamfer distance for whole scene
python chf_loss_allseq.py
# chamfer distance for car, person and cyclist
python ins_chf_loss_allseq.py
# voxel IoU for whole scene
python voxel_iou_allseq.py
# voxel IoU for car, person and cyclist
python ins_voxel_iou_allseq
~~~

## Result comparison
![image](https://github.com/willy8898/INS_ATT/assets/62001022/51e1b8bf-5539-4de2-bc04-7550379e68cd)










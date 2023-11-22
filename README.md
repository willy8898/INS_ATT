# INS_ATT

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
![image](https://github.com/willy8898/INS_ATT/assets/62001022/a09e52a3-e8cd-4ffc-9b3a-2758542cdb9d =300x)

## Dataset preparation
You should specify the folder contains the original [Semantic-KITTI](http://www.semantic-kitti.org/dataset.html) dataset and the processed dataset folder are.
~~~
cd dataset_prepare
python scene_preprocess.py
~~~

## Training
~~~
python train.py
~~~

## Inference
~~~
cd inference
python infer_kitti_ins_allseq
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






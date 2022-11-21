# DH-AUG-DH-Forward-Kinematics-Model-Driven-Augmentation-for-3D-Human-Pose-Estimation
Code repository for the paper:

**DH-AUG: DH Forward Kinematics Model Driven Augmentation for 3D Human Pose Estimation**

Linzhi Huang, Jiahao Liang,Weihong Deng*

ECCV 2022 

[[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660427.pdf)]

## Dataset setup
### [Human3.6M](http://vision.imar.ro/human3.6m/)
The code for Human3.6M data preparation is borrowed from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D), [SemGCN](https://github.com/garyzhao/SemGCN), [EvoSkeleton](https://github.com/Nicholasli1995/EvoSkeleton), [PoseAug](https://github.com/jfzhang95/PoseAug).

### Prepare the ground truth 2D 3D data pair for Human3.6
* Setup from original source (recommended)
    * Please follow the instruction from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md) to process the data from the official [Human3.6M](http://vision.imar.ro/human3.6m/) website.
    * Then generate the 2D and 3D data by `prepare_data_h36m.py`. (Note that `prepare_data_h36m.py` is borrowed from [SemGCN](https://github.com/garyzhao/SemGCN) with 16 joints configuration, which is slightly different from [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) with 17 joints configuration)
    
* Setup from preprocessed dataset
    * Get preprocessed `h36m.zip`: 
      Please follow the instruction from [SemGCN](https://github.com/garyzhao/SemGCN/blob/master/data/README.md) to get the `h36m.zip`.
    * Convert `h36m.zip` to ground-truth 2D 3D npz file: 
      Process `h36m.zip` by `prepare_data_h36m.py` to get `data_3d_h36m.npz` and `data_2d_h36m_gt.npz`
```sh
cd data
python prepare_data_h36m.py --from-archive h36m.zip
cd ..
```
After this step, you should end up with two files in the `data` directory: `data_3d_h36m.npz` for the 3D poses, and `data_2d_h36m_gt.npz` for the ground-truth 2D poses,
which will look like:
   ```
   ${DH-AUG}
   ├── data
      ├── data_3d_h36m.npz
      ├── data_2d_h36m_gt.npz
   ```

### Prepare other detected 2D pose for Human3.6M (optional)
In this step, you need to download the detected 2D pose npz file and delete the Neck/Nose axis (e.g., the shape of array: nx17x2 -> nx16x2; n: number_of_frame) for every subject and action.

* To download the Det and CPN 2D pose, please follow the instruction of [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md) and download the `data_2d_h36m_cpn_ft_h36m_dbb.npz` and `data_2d_h36m_detectron_ft_h36m.npz`. 

```sh
cd data
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_cpn_ft_h36m_dbb.npz
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_detectron_ft_h36m.npz
cd ..
``` 

* To download the HHR 2D pose, please follow the instruction of [EvoSkeleton](https://github.com/Nicholasli1995/EvoSkeleton/blob/master/docs/HHR.md) and download the `twoDPose_HRN_test.npy` and `twoDPose_HRN_train.npy`, 
and convert them to the same format as `data_2d_h36m_gt.npz`.

* You can also download [poseAug pre-processed joints files](https://drive.google.com/drive/folders/1jVyz9HdT0Jq3-YPZnOQ6GEcOVDRZAifK?usp=sharing). 

Until here, you will have a data folder:
   ```
   ${DH-AUG}
   ├── data
      ├── data_3d_h36m.npz
      ├── data_2d_h36m_gt.npz
      ├── data_2d_h36m_detectron_ft_h36m.npz
      ├── data_2d_h36m_cpn_ft_h36m_dbb.npz
      ├── data_2d_h36m_hr.npz
   ```
Please make sure the 2D data are all 16 joints setting.


### [3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)
The code for 3DHP data preparation is borrowed from [SPIN](https://github.com/nkolot/SPIN)

* Please follow the instruction from [SPIN](https://github.com/nkolot/SPIN/blob/master/fetch_data.sh) to download the preprocessed compression file `dataset_extras.tar.gz` then unzip it to get mpi_inf_3dhp_valid.npz and put it at `data_extra/dataset_extras/mpi_inf_3dhp_valid.npz` (24 joints).
* Then process the `dataset_extras/mpi_inf_3dhp_valid.npz` with `prepare_data_3dhp.py` or `prepare_data_3dhp.ipynb` file to get the `test_3dhp.npz` (16 joints) and place it at `data_extra/test_set`.

Until here, you will have a data_extra folder:
   ```
   ${DH-AUG}
   ├── data_extra
      ├── bone_length_npy
         ├── hm36s15678_bl_templates.npy
      ├── dataset_extras
         ├── mpi_inf_3dhp_valid.npz
         ├── ... (not in use)
      ├── test_set
         ├── test_3dhp.npz
      ├── prepare_data_3dhp.ipynb
      ├── prepare_data_3dhp.py
   ```

All the data are set up.



##  Training   
Examples of training the baseline model with DH-Aug:
If you want to explore better performance for specific setting, please try changing the hyper-param.
### VPose (single-frame)
python3 run_Fk_GAN.py --note posefk --posenet_name 'videopose' --lr_p 1e-4  --checkpoint './checkpoint/posefk' --keypoints gt --s1only False --GAN_whether_use_preAngle True  --video_over_200mm False --batch_size 1024  --data_enhancement_method 'GAN'  --additional_LR_decay 0.95 --Gen_DenseDim 256 --Dis_DenseDim_3D 256  --Dis_DenseDim_2D  256

### VPose (video)
 python3 run_Fk_GAN.py --note posefk --posenet_name 'mulit_farme_videopose' --lr_p 1e-3 --checkpoint './checkpoint/posefk' --keypoints gt --s1only False --GAN_whether_use_preAngle True  --single_or_multi_train_mode multi --video_over_200mm False --batch_size 512  --data_enhancement_method 'GAN' --downsample 10 --additional_LR_decay 0.95 --warmup  20 --single_dis_warmup_epoch 4 --architecture '3,3'

## DH-3DHP dataset 
url：https://pan.baidu.com/s/1o94c5Gwt7votJSw7mY5o_A 
Extraction code：hlz1

## Citation
If you  find this code useful for your research, please consider citing the following paper:

    @inproceedings{Huang2022DH-AUG,
      title       = {DH-AUG: DH Forward Kinematics Model Driven Augmentation for 3D Human Pose Estimation},
      author      = {Linzhi Huang, Jiahao Liang, Weihong Deng},
      booktitle   = {ECCV},
      year        = {2022}
    }

## Acknowledgements
This code uses ([SemGCN](https://github.com/garyzhao/SemGCN), [SimpleBL](https://github.com/una-dinosauria/3d-pose-baseline), [PoseAUG](https://github.com/jfzhang95/PoseAug), [PoseFormer](https://github.com/zczcwh/PoseFormer) and [VPose3D](https://github.com/facebookresearch/VideoPose3D)) as backbone. We gratefully appreciate the impact these libraries had on our work. If you use our code, please consider citing the original papers as well.


# DH-AUG-DH-Forward-Kinematics-Model-Driven-Augmentation-for-3D-Human-Pose-Estimation
Code repository for the paper:
**DH-AUG: DH Forward Kinematics Model Driven Augmentation for 3D Human Pose Estimation**
Linzhi Huang, Jiahao Liang,Weihong Deng*
ECCV 2022 
[paper] [project page (coming soon)]

## Prepare dataset
# [Human3.6M]
The code for Human3.6M data preparation is borrowed from [VideoPose3D], [SemGCN], [EvoSkeleton].


* Setup from preprocessed dataset
    * Get preprocessed `h36m.zip`: 
      Please follow the instruction from [SemGCN] to get the `h36m.zip`.
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
   ├── data
      ├── data_3d_h36m.npz
      ├── data_2d_h36m_gt.npz
   ```


# [3DHP]
The code for 3DHP data preparation is borrowed from [SPIN]

* Please follow the instruction from [SPIN] to download the preprocessed compression file `dataset_extras.tar.gz` then unzip it to get mpi_inf_3dhp_valid.npz and put it at `data_extra/dataset_extras/mpi_inf_3dhp_valid.npz` (24 joints).
* Then process the `dataset_extras/mpi_inf_3dhp_valid.npz` with `prepare_data_3dhp.py` or `prepare_data_3dhp.ipynb` file to get the `test_3dhp.npz` (16 joints) and place it at `data_extra/test_set`.

Until here, you will have a data_extra folder:
   ```
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


##  training   



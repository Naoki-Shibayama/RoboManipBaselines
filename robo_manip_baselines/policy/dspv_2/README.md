# DSPv2

## Install
See [here](../../../doc/install.md#DSPv2) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

## Data preprocessing
See [here](../../../doc/preprocessing_pointcloud.md) to perform data preprocessing for 3D point clouds.

> [!NOTE]
> In generate and store pointcloud step, set image_size=[224, 224] like following command:
> ```console
> $ python ./misc/AddPointCloudToRmbData.py ./dataset/<dataset_name> --min_bound <x, y, z> --max_bound <x, y, z> --rpy_angle <roll, pitch, yaw> --image_size 224 224
> ```

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py Dspv2 --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/Dspv2/<checkpoint_name>
```

## Policy rollout
Run a trained policy:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Rollout.py Dspv2 MujocoUR5eCable --checkpoint ./checkpoint/Dspv2/<checkpoint_name>/policy_last.ckpt
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@misc{su2025dspv2improveddensepolicy,
    title={DSPv2: Improved Dense Policy for Effective and Generalizable Whole-body Mobile Manipulation},
    author={Yue Su and Chubin Zhang and Sijin Chen and Liufan Tan and Yansong Tang and Jianan Wang and Xihui Liu},
    year={2025},
    eprint={2509.16063},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2509.16063},
}
```

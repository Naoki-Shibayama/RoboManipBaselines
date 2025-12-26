# Diffusion Policy With AnyTouch

## Install
See [here](../../../doc/install.md#Diffusion-policy-with-AnyTouch) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop). RGB tactile sensor data is required.

> [!NOTE]
> If you are using `pyenv` and encounter the error `No module named '_bz2'`, apply the following solution.  
> https://stackoverflow.com/a/71457141

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py DiffusionPolicyWithAnyTouch --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/DiffusionPolicyWithAnyTouch/<checkpoint_name> --anytouch_checkpoint ./common/models/checkpoints/anytouch.pth --anytouch_config ./common/models/configs/AnyTouchConfig.json
```

> [!NOTE]
> If you encounter the following error,
> ```console
> ImportError: cannot import name 'cached_download' from 'huggingface_hub'
> ```
> downgrade `huggingface_hub` by the following command.
> ```console
> $ pip install huggingface_hub==0.24.6
> ```

## Policy rollout
Run a trained policy:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Rollout.py DiffusionPolicyWithAnyTouch MujocoUR5eCable --checkpoint ./checkpoint/DiffusionPolicyWithAnyTouch/<checkpoint_name>/policy_last.ckpt
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@INPROCEEDINGS{DiffusionPolicy_RSS23,,
  author = {Chi, Cheng and Feng, Siyuan and Du, Yilun and Xu, Zhenjia and Cousineau, Eric and Burchfiel, Benjamin and Song, Shuran},
  title = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
  booktitle = {Proceedings of Robotics: Science and Systems},
  year = {2023},
  month = {July},
  doi = {10.15607/RSS.2023.XIX.026}
},
@inproceedings{fenganytouch,
  	title={AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors},
  	author={Feng, Ruoxuan and Hu, Jiangyu and Xia, Wenke and Shen, Ao and Sun, Yuhao and Fang, Bin and Hu, Di and others},
  	booktitle={The Thirteenth International Conference on Learning Representations}
}
```

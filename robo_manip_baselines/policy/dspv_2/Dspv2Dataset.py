import os
import sys

import cv2
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/DSPv2"))


from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    get_skipped_data_seq,
    get_skipped_single_data,
    normalize_data,
)
from robo_manip_baselines.common.utils.Vision3dUtils import (
    voxelize_pointcloud_for_dspv2,
)


def dspv2_collate_fn(batch):
    feats, coords, action, state, v_image = list(zip(*batch))

    action_b = torch.stack(action, dim=0)
    state_b = torch.stack(state, dim=0)
    v_image_b = torch.tensor(
        np.array([image.numpy() for image in v_image]), dtype=torch.float32
    )

    return coords, feats, action_b, state_b, v_image_b


class Dspv2Dataset(DatasetBase):
    """Dataset to train 3D diffusion policy."""

    def setup_variables(self):
        self.chunk_info_list = []
        skip = self.model_meta_info["data"]["skip"]
        n_action_steps = self.model_meta_info["data"]["n_action_steps"]
        self.voxel_size = self.model_meta_info["data"]["voxel_size"]
        self.n_pointcloud_dim = 6 if self.model_meta_info["data"]["use_pc_color"] else 3
        pad_after = self.model_meta_info["data"]["n_action_steps"] - 1

        for episode_idx, filename in enumerate(self.filenames):
            with RmbData(filename) as rmb_data:
                episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
                for start_time_idx in range(
                    0, episode_len - (n_action_steps - 1) + pad_after
                ):
                    self.chunk_info_list.append((episode_idx, start_time_idx))

    def __len__(self):
        return len(self.chunk_info_list)

    def __getitem__(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        n_action_steps = self.model_meta_info["data"]["n_action_steps"]
        episode_idx, start_time_idx = self.chunk_info_list[chunk_idx]

        with RmbData(self.filenames[episode_idx], self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            action_time_idxes = np.clip(
                np.arange(start_time_idx, start_time_idx + n_action_steps),
                0,
                episode_len - 1,
            )

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        get_skipped_single_data(
                            rmb_data[key][:], start_time_idx, key, skip
                        )
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_data_seq(rmb_data[key][:], key, skip)[action_time_idxes]
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            # Load pointcloud
            camera_name = self.model_meta_info["image"]["camera_names"][0]
            pointcloud = rmb_data[DataKey.get_pointcloud_key(camera_name)][::skip][
                start_time_idx
            ]

            # Load view images
            v_images = np.stack(
                [
                    rmb_data[DataKey.get_rgb_image_key(camera_name)][::skip][
                        start_time_idx
                    ]
                    for camera_name in self.model_meta_info["image"][
                        "view_camera_names"
                    ]
                ],
                axis=0,
            )

        # Resize view images
        K, H, W, C = v_images.shape
        image_size = self.model_meta_info["data"]["image_size"]
        v_images = np.array(
            [cv2.resize(img, image_size) for img in v_images.reshape(-1, H, W, C)]
        ).reshape(K, 1, *image_size[::-1], C)

        # Pre-convert data
        state, action, pcd_coords, pcd_feats, v_images = self.pre_convert_data(
            state, action, pointcloud, v_images
        )
        v_images = v_images.reshape(K, C, *image_size[::-1])

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        pcd_coords_tensor = torch.tensor(pcd_coords, dtype=torch.float32)
        pcd_feats_tensor = torch.tensor(pcd_feats, dtype=torch.float32)
        v_image_tensor = torch.tensor(v_images, dtype=torch.uint8)

        # Augment data
        state_tensor, action_tensor, v_image_tensor = self.augment_data(
            state_tensor, action_tensor, v_image_tensor
        )

        # Sort in the order of policy inputs and outputs
        return (
            pcd_feats_tensor,
            pcd_coords_tensor,
            action_tensor,
            state_tensor,
            v_image_tensor,
        )

    def pre_convert_data(self, state, action, pointcloud, images):
        """Pre-convert data. Arguments must be numpy arrays (not torch tensors)."""
        state, action, images = super().pre_convert_data(state, action, images)
        pointcloud_coords, pointcloud = voxelize_pointcloud_for_dspv2(
            pointcloud, self.voxel_size
        )
        pointcloud_feats = normalize_data(
            pointcloud, self.model_meta_info["pointcloud"]
        )[:, : self.n_pointcloud_dim]

        return state, action, pointcloud_coords, pointcloud_feats, images

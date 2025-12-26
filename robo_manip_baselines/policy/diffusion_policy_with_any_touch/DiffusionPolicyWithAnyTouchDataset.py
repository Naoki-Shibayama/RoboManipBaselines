import cv2
import numpy as np
import torch
from torchvision.transforms import v2

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    DpStyleDatasetMixin,
    RmbData,
    get_skipped_data_seq,
)


class DiffusionPolicyWithAnyTouchDataset(DatasetBase, DpStyleDatasetMixin):
    """Dataset to train diffusion policy."""

    def setup_variables(self):
        self.setup_dp_style_chunk()
        self.sensor_type_map = {
            "gelsight": 0,
            "digit": 1,
            "gelslim": 2,
            "gelsight_mini": 3,
            "duragel": 4,
        }
        self.use_universal_sensor_token = self.model_meta_info["data"][
            "use_universal_sensor_token"
        ]

    def setup_image_transforms(self):
        super().setup_image_transforms()

        self.tactile_image_transforms = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.3),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.chunk_info_list)

    def __getitem__(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        episode_idx, start_time_idx = self.chunk_info_list[chunk_idx]

        with RmbData(self.filenames[episode_idx], self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            time_idxes = np.clip(
                np.arange(start_time_idx, start_time_idx + horizon), 0, episode_len - 1
            )

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes]
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                    axis=1,
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes]
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            # Load images
            images = np.stack(
                [
                    rmb_data[DataKey.get_rgb_image_key(camera_name)][::skip][time_idxes]
                    for camera_name in self.model_meta_info["image"]["camera_names"]
                ],
                axis=0,
            )

            # Load tactile images
            tactile_images = np.stack(
                [
                    rmb_data[DataKey.get_rgb_image_key(camera_name)][::skip][time_idxes]
                    for camera_name in self.model_meta_info["data"][
                        "tactile_camera_names"
                    ]
                ],
                axis=0,
            )

        # Resize images
        K, T, H, W, C = images.shape
        image_size = self.model_meta_info["data"]["image_size"]
        images = np.array(
            [cv2.resize(img, image_size) for img in images.reshape(-1, H, W, C)]
        ).reshape(K, T, *image_size[::-1], C)
        # Resize tactile images
        K, T, H, W, C = tactile_images.shape
        image_size = (224, 224)
        tactile_images = np.array(
            [cv2.resize(img, image_size) for img in tactile_images.reshape(-1, H, W, C)]
        ).reshape(K, T, *image_size[::-1], C)

        # Pre-convert data
        state, action, images = self.pre_convert_data(state, action, images)
        _, _, tactile_images = self.pre_convert_data(None, None, tactile_images)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        images_tensor = torch.tensor(images, dtype=torch.uint8)
        tactile_images_tensor = torch.tensor(tactile_images, dtype=torch.uint8)

        # Augment data
        state_tensor, action_tensor, images_tensor, tactile_images_tensor = (
            self.augment_data(
                state_tensor, action_tensor, images_tensor, tactile_images_tensor
            )
        )

        # Convert to data structure of policy input and output
        data = {
            "obs": {"image": {}, "tactile_image": {}, "tactile_sensor_type": {}},
            "action": action_tensor,
            "tactile": {},
        }
        if len(self.model_meta_info["state"]["keys"]) > 0:
            data["obs"]["state"] = state_tensor
        for camera_idx, camera_name in enumerate(
            self.model_meta_info["image"]["camera_names"]
        ):
            data["obs"]["image"][DataKey.get_rgb_image_key(camera_name)] = (
                images_tensor[camera_idx]
            )
        for camera_idx, camera_name in enumerate(
            self.model_meta_info["data"]["tactile_camera_names"]
        ):
            data["obs"]["tactile_image"][DataKey.get_rgb_image_key(camera_name)] = (
                tactile_images_tensor[camera_idx]
            )
            if self.use_universal_sensor_token:
                data["obs"]["tactile_sensor_type"][camera_name] = -1
            else:
                data["obs"]["tactile_sensor_type"][camera_name] = (
                    self.get_sensor_type_from_camera_name(camera_name)
                )

        return data

    def get_sensor_type_from_camera_name(self, camera_name):
        for k in self.sensor_type_map.keys():
            if k in camera_name:
                if k != "gelsight":
                    return self.sensor_type_map[k]
                elif "gelsight_mini" in camera_name:
                    return self.sensor_type_map["gelsight_mini"]
                else:
                    return self.sensor_type_map[k]
        return 5  # unkown sensor

    def augment_data(self, state, action, images, tactile_images):
        state, action, images = super().augment_data(state, action, images)

        # Adjust to a range from -1 to 1 to match the original implementation
        images = images * 2.0 - 1.0

        if tactile_images is not None:
            if tactile_images.ndim < 3:
                raise ValueError(
                    f"[{self.__class__.__name__}] Input must have at least 3 dimensions (C, H, W)"
                )
            elif tactile_images.ndim == 3:
                tactile_images = self.tactile_image_transforms(tactile_images)
            else:
                orig_prefix_shape = tactile_images.shape[:-3]
                orig_image_list = tactile_images.reshape(-1, *tactile_images.shape[-3:])
                transformed_image_list = [
                    self.tactile_image_transforms(img) for img in orig_image_list
                ]
                tactile_images = torch.stack(transformed_image_list)
                new_img_shape = tactile_images.shape[-3:]
                tactile_images = tactile_images.view(*orig_prefix_shape, *new_img_shape)

        return state, action, images, tactile_images

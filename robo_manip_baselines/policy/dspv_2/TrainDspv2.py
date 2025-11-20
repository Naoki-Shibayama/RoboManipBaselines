import os
import sys

import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/DSPv2"))
import MinkowskiEngine as ME
from policy.policy import dspv2

from robo_manip_baselines.common import CachedDataset, TrainBase, TrainPointCloudMixin

from .Dspv2Dataset import Dspv2Dataset, dspv2_collate_fn


class TrainDspv2(TrainBase, TrainPointCloudMixin):
    DatasetClass = Dspv2Dataset

    def make_dataloader(self, filenames, shuffle=True):
        dataset = self.DatasetClass(
            filenames, self.model_meta_info, self.args.enable_rmb_cache
        )
        if self.args.use_cached_dataset:
            dataset = CachedDataset(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.args.num_workers,
            persistent_workers=True,
            prefetch_factor=4,
            collate_fn=dspv2_collate_fn,
        )

        return dataloader

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(norm_type="limits")

        parser.set_defaults(batch_size=32)
        parser.set_defaults(num_epochs=1000)
        parser.set_defaults(lr=3e-4)

        parser.set_defaults(
            state_keys=["measured_eef_pose", "measured_gripper_joint_pos"]
        )
        parser.set_defaults(
            action_keys=["command_eef_pose", "command_gripper_joint_pos"]
        )
        parser.add_argument(
            "--n_action_steps",
            type=int,
            default=16,
            help="number of steps in the action sequence to output from the policy",
        )
        parser.add_argument(
            "--hidden_dim", type=int, default=512, help="hidden dimension"
        )
        parser.add_argument(
            "--dim_feedforward", type=int, default=2048, help="feedforward dimension"
        )
        parser.add_argument(
            "--use_pc_color",
            action="store_true",
            help="Whether to use color information of point cloud",
        )
        parser.add_argument(
            "--dino_version",
            type=str,
            default="v2",
            choices=["v2", "v3"],
            help="DINO version (v3 requires HuggingFace account and agreement about sharing your contact information)",
        )
        parser.add_argument(
            "--view_camera_names",
            type=str,
            nargs="+",
            default=["front", "hand", "left", "right"],
            help="camera names to use as view",
        )

    def set_data_stats(self):
        super().set_data_stats()

        self.set_pointcloud_stats()

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["image"]["view_camera_names"] = self.args.view_camera_names
        self.model_meta_info["data"]["n_action_steps"] = self.args.n_action_steps
        self.model_meta_info["data"]["use_pc_color"] = self.args.use_pc_color

        num_points, image_size, min_bound, max_bound, rpy_angle = (
            self.setup_pointcloud_info()
        )
        self.model_meta_info["data"]["num_points"] = num_points
        self.model_meta_info["data"]["image_size"] = image_size
        self.model_meta_info["data"]["min_bound"] = min_bound
        self.model_meta_info["data"]["max_bound"] = max_bound
        self.model_meta_info["data"]["rpy_angle"] = rpy_angle

    def setup_policy(self):
        # Set policy args
        self.model_meta_info["policy"]["args"] = {
            "Tp": self.args.n_action_steps,
            "Ta": self.args.n_action_steps,
            "input_dim": 6 if self.args.use_pc_color else 3,
            "obs_feature_dim": 512,
            "action_dim": len(self.model_meta_info["action"]["example"]),
            "hidden_dim": self.args.hidden_dim,
            "dim_feedforward": self.args.dim_feedforward,
            "num_encoder_layers": 4,
            "num_decoder_layers": 1,
            "nheads": 8,
            "num_views": len(self.args.view_camera_names),
            "dino_version": self.args.dino_version,
        }

        # Construct policy
        self.policy = dspv2(**self.model_meta_info["policy"]["args"])
        self.policy.cuda()

        # Construct optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.args.lr,
            betas=[0.95, 0.999],
            weight_decay=1e-6,
        )
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=2000,
            num_training_steps=(len(self.train_dataloader) * self.args.num_epochs),
        )

        # Print policy information
        self.print_policy_info()
        print(
            f"  - dino version: {self.args.dino_version}, view_camera_names: {self.args.view_camera_names}"
        )

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                pcd_feats, pcd_coords, action, state, v_image = [d for d in data]
                pcd_coords, pcd_feats = ME.utils.sparse_collate(pcd_coords, pcd_feats)
                pcd = ME.SparseTensor(pcd_feats.cuda(), pcd_coords.cuda())
                loss = self.policy(pcd, action.cuda(), state.cuda(), v_image.cuda())
                loss.backward()
                self.optimizer.step()
                batch_result_list.append(self.detach_batch_result({"loss": loss}))
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Run validation step
            with torch.inference_mode():
                self.policy.eval()
                batch_result_list = []
                for data in self.val_dataloader:
                    pcd_feats, pcd_coords, action, state, v_image = [d for d in data]
                    pcd_coords, pcd_feats = ME.utils.sparse_collate(
                        pcd_coords, pcd_feats
                    )
                    pcd = ME.SparseTensor(pcd_feats.cuda(), pcd_coords.cuda())
                    loss = self.policy(pcd, action.cuda(), state.cuda(), v_image.cuda())
                    batch_result_list.append(self.detach_batch_result({"loss": loss}))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                self.update_best_ckpt(epoch_summary)

            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>3}")

        # Save last checkpoint
        self.save_current_ckpt("last")

        # Save best checkpoint
        self.save_best_ckpt()

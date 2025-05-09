from os import path

import numpy as np

from .MujocoUR5eEnvBase import MujocoUR5eEnvBase


class MujocoUR5eClothEnv(MujocoUR5eEnvBase):
    default_camera_config = {
        "azimuth": 45.0,
        "elevation": -45.0,
        "distance": 1.8,
        "lookat": [-0.2, -0.2, 0.8],
    }

    def __init__(
        self,
        **kwargs,
    ):
        MujocoUR5eEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/ur5e/env_ur5e_cloth.xml",
            ),
            np.array(
                [
                    1.2 * np.pi,
                    -np.pi / 2,
                    -0.85 * np.pi,
                    -0.65 * np.pi,
                    -0.5 * np.pi,
                    0.0,
                    *np.zeros(8),
                ]
            ),
            **kwargs,
        )

        self.original_cloth_pos = self.model.body("cloth").pos.copy()
        self.original_board_pos = self.model.body("board").pos.copy()
        self.pos_cloth_offsets = np.array(
            [
                [0.0, -0.12, 0.0],
                [0.0, -0.08, 0.0],
                [0.0, -0.04, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.04, 0.0],
                [0.0, 0.08, 0.0],
            ]
        )  # [m]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.pos_cloth_offsets)
        self.model.body("cloth").pos = (
            self.original_cloth_pos + self.pos_cloth_offsets[world_idx]
        )
        self.model.body("board").pos = (
            self.original_board_pos + self.pos_cloth_offsets[world_idx]
        )
        return world_idx

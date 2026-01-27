from os import path

import numpy as np
import pybulletX as px

from .TactoSawyerEnvBase import Camera, TactoSawyerEnvBase


class TactoSawyerSpoonEnv(TactoSawyerEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        TactoSawyerEnvBase.__init__(
            self,
            np.array(
                [
                    0,
                    -0.9,
                    -0.45,
                    1.4,
                    0.27,
                    1.2,
                    1.2,
                    -0.02,
                    0.02,
                ]
            ),
        )

        self.cameras["front"] = Camera(
            camera_pos=[0.8, 0, 0.05],
            camera_distance=1.0,
            yaw=90,
            pitch=-30.0,
            roll=0,
        )

        self.start_obj_pos_offsets = np.array(
            [
                [-0.3, 0, 0],
                [-0.2, 0, 0],
                [-0.1, 0, 0],
                [0, 0, 0],
                [0.1, 0, 0],
                [0.2, 0, 0],
            ]
        )

    def setup_task_specific_object(self):
        self.box_scaling = 0.4
        self.spoon = px.Body(
            urdf_path=path.join(
                path.dirname(__file__), "../assets/tacto/objects/spoon/dummy_spoon.urdf"
            ),
            base_position=[0.7, 0, 0.2],
            base_orientation=[0, -0.5, 0, 1],
            global_scaling=0.2,
        )
        self.start_box = px.Body(
            urdf_path=path.join(
                path.dirname(__file__), "../assets/tacto/objects/box/white_box.urdf"
            ),
            base_position=[0.8, 0, 0.08],
            global_scaling=self.box_scaling,
        )
        self.green_box = px.Body(
            urdf_path=path.join(
                path.dirname(__file__), "../assets/tacto/objects/box/green_box.urdf"
            ),
            base_position=[0.8, -0.5, 0.08],
            global_scaling=self.box_scaling,
        )
        self.blue_box = px.Body(
            urdf_path=path.join(
                path.dirname(__file__), "../assets/tacto/objects/box/blue_box.urdf"
            ),
            base_position=[0.8, 0.5, 0.08],
            global_scaling=self.box_scaling,
        )

        self.goal_boxes = [self.green_box, self.blue_box]
        self.all_task_obj = [self.spoon, self.start_box, self.green_box, self.blue_box]
        for obj in self.all_task_obj:
            self.rgb_tactiles.add_body(obj)

    def reset_task_specific_object(self):
        for obj in self.goal_boxes:
            obj.reset()

    def _get_reward(self):
        (x, y, z), _ = self.spoon.get_base_pose()
        goal_height = 0.3 * self.box_scaling
        for box in self.goal_boxes:
            (bx, by, _), _ = box.get_base_pose()
            bx_min = bx - (0.225 * self.box_scaling)
            bx_max = bx + (0.225 * self.box_scaling)
            by_min = by - (0.225 * self.box_scaling)
            by_max = by + (0.225 * self.box_scaling)
            if (
                z < goal_height
                and (bx_min < x and x < bx_max)
                and (by_min < y and y < by_max)
            ):
                return 1.0
        return 0.0

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        if world_idx is None:
            world_idx = cumulative_idx % len(self.start_obj_pos_offsets)

        if self.world_random_scale is not None:
            rand_offset = np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        else:
            rand_offset = np.zeros(3)

        for obj in [self.spoon, self.start_box]:
            pos = obj.init_base_position.copy()
            pos += self.start_obj_pos_offsets[world_idx] + rand_offset
            obj.set_base_pose(pos, obj.init_base_orientation)

        return world_idx

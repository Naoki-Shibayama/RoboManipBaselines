from os import path

import numpy as np
import pybulletX as px

from .TactoSawyerEnvBase import TactoSawyerEnvBase


class TactoSawyerGraspEnv(TactoSawyerEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        TactoSawyerEnvBase.__init__(
            self,
            np.array(
                [
                    0,
                    -1.24,
                    -0.58,
                    2.18,
                    0.27,
                    0.7,
                    1,
                    -0.005,
                    0.005,
                ]
            ),
        )

    def setup_task_specific_object(self):
        self.obj = px.Body(
            urdf_path=path.join(
                path.dirname(__file__), "../assets/tacto/objects/cube/cube_small.urdf"
            ),
            base_position=[0.50, 0, 0.02],
            global_scaling=0.6,
        )

        self.rgb_tactiles.add_body(self.obj)

    def reset_task_specific_object(self):
        self.obj.reset()
        self.modify_world()

    def _get_reward(self):
        (x, y, z), _ = self.obj.get_base_pose()
        linear_velocity, angular_velocity = self.obj.get_base_velocity()
        linear_velocity = np.linalg.norm(linear_velocity)
        angular_velocity = np.linalg.norm(angular_velocity)

        if z > 0.1 and linear_velocity < 0.025 and angular_velocity < 0.025:
            return 1.0
        else:
            return 0.0

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        pos = self.obj.init_base_position.copy()
        if self.world_random_scale is not None:
            pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        self.obj.set_base_pose(pos)

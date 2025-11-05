import numpy as np

from .RealUR5eDualEnvBase import RealUR5eDualEnvBase


class RealUR5eDualDemoEnv(RealUR5eDualEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        RealUR5eDualEnvBase.__init__(
            self,
            init_qpos=np.array(
                [
                    # left arm from workspace side
                    -1.18000162,
                    -1.22462273,
                    -1.5561803,
                    -1.9295612,
                    1.57465679,
                    0.39695961,
                    0.0,
                    # right arm from workspace side
                    1.18000162,
                    -1.91696992,
                    1.5561803,
                    -1.21203147,
                    -1.57465679,
                    -0.39695961,
                    0.0,
                ]
            ),
            **kwargs,
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        # TODO: Automatically set world index according to task variations
        if world_idx is None:
            world_idx = 0
            # world_idx = cumulative_idx % 2
        return world_idx

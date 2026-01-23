import time
from abc import ABC, abstractmethod
from os import path

import gymnasium as gym
import numpy as np
import pybullet as p
import pybulletX as px
from gymnasium.spaces import Dict
from tacto import Sensor

from robo_manip_baselines.common import ArmConfig, DataKey, EnvDataMixin
from robo_manip_baselines.teleop import (
    GelloInputDevice,
    KeyboardInputDevice,
    SpacemouseInputDevice,
)


class Camera:
    def __init__(
        self,
        camera_pos,
        camera_distance,
        yaw,
        pitch,
        roll,
        width=640,
        height=480,
        fov=60,
        near_plane=0.01,
        far_plane=100.0,
        up_axis_index=2,
    ):
        self.width = width
        self.height = height

        camTargetPos = camera_pos
        camDistance = camera_distance
        upAxisIndex = up_axis_index

        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll

        self.fov = fov
        nearPlane = near_plane
        farPlane = far_plane

        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll(
            camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex
        )

        aspect = width / height

        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov, aspect, nearPlane, farPlane
        )

    def get_image(self):
        img_arr = p.getCameraImage(
            self.width,
            self.height,
            self.viewMatrix,
            self.projectionMatrix,
            shadow=1,
            lightDirection=[1, 1, 1],
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb = img_arr[2][:, :, :3]  # color image RGB H x W x 3 (uint8)
        dep = img_arr[3]  # depth image H x W (float32)
        return rgb, dep


class TactoSawyerEnvBase(EnvDataMixin, gym.Env, ABC):
    metadata = {
        "render_modes": [
            "human",
        ],
    }

    def __init__(
        self,
        init_qpos,
        **kwargs,
    ):
        self.init_time = (
            time.time()
        )  # TODO: timeは使わずにシミュレーションの時刻を用いる
        self.init_qpos = init_qpos
        self.render_mode = kwargs.get("render_mode")

        # Setup tacto
        px.init(mode=p.GUI)

        self.setup_robot(init_qpos)
        self.setup_rgb_tactile_sensor()
        self.setup_camera()
        self.setup_task_specific_object()

        self.world_random_scale = None
        self.reset()

        # Setup environment parameters
        self.dt = 0.02  # [s]

        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.dt)

        self.action_spacekey = (
            "joint_torque" if self.robot.torque_control else "joint_position"
        )
        self.action_space = self.robot.action_space[self.action_spacekey]
        self.observation_space = Dict(
            {
                "joint_pos": self.robot.state_space["joint_position"],
                "joint_vel": self.robot.state_space["joint_velocity"],
                "wrench": self.robot.state_space[
                    "joint_reaction_forces"
                ],  # TODO: 全関節の6軸トルクのspaceになっていないか要確認
            }
        )

    def setup_robot(self, init_qpos):
        self.robot = px.Robot(
            urdf_path=path.join(
                path.dirname(__file__),
                "../assets/tacto/robots/sawyer/sawyer_wsg50.urdf",
            ),
            use_fixed_base=True,
        )
        self.robot.zero_pose = init_qpos
        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__), "../assets/common/robots/sawyer/sawyer.urdf"
                ),
                arm_root_pose=self.get_link_pose("right_j0"),
                ik_eef_joint_id=7,
                arm_joint_idxes=np.arange(7),
                gripper_joint_idxes=np.array([7]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[0:7],
                init_gripper_joint_pos=np.array([0.005]),
            )
        ]
        p.enableJointForceTorqueSensor(
            self.robot.id, self.robot.get_joint_index_by_name("right_hand"), True
        )
        self.robot.configure_state_space(applied_joint_motor_torque=False)

    def setup_rgb_tactile_sensor(self):
        self.rgb_tactiles = Sensor(width=640, height=480, visualize_gui=False)
        tactile_joint_names = ["joint_finger_tip_left", "joint_finger_tip_right"]
        tactile_link_ids = [
            self.robot.get_joint_index_by_name(name) for name in tactile_joint_names
        ]
        self.rgb_tactiles.add_camera(self.robot.id, tactile_link_ids)

    def setup_camera(self):
        self.cameras = {}
        self.cameras["front"] = Camera(
            camera_pos=[0.5, 0, 0.05],
            camera_distance=0.4,
            yaw=90,
            pitch=-30.0,
            roll=0,
        )

    @abstractmethod
    def setup_task_specific_object(self):
        pass

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.robot.reset()

        self.reset_task_specific_object()

        self.obs = self._get_obs()
        self.info = self._get_info()

        return self.obs, self.info

    @abstractmethod
    def reset_task_specific_object(self):
        pass

    def setup_input_device(self, input_device_name, motion_manager, overwrite_kwargs):
        if input_device_name == "spacemouse":
            InputDeviceClass = SpacemouseInputDevice
        elif input_device_name == "gello":
            InputDeviceClass = GelloInputDevice
        elif input_device_name == "keyboard":
            InputDeviceClass = KeyboardInputDevice
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid input device key: {input_device_name}"
            )

        default_kwargs = self.get_input_device_kwargs(input_device_name)

        return [
            InputDeviceClass(
                motion_manager.body_manager_list[0],
                **{**default_kwargs, **overwrite_kwargs},
            )
        ]

    def get_input_device_kwargs(self, input_device_name):
        return {"gripper_scale": 0.005}

    def step(self, action):
        self._set_actions(action)
        p.stepSimulation()

        self.obs = self._get_obs()
        reward = self._get_reward()
        terminated = False
        self.info = self._get_info()

        # Update viewer
        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return (self.obs, reward, terminated, False, self.info)

    def _set_actions(self, action):
        gripper_joint_pos = action[-1]
        tacto_action = np.concatenate(
            (action[:7], np.array([gripper_joint_pos, -1 * gripper_joint_pos]))
        )
        if self.robot.torque_control:
            self.robot.set_joint_torque(tacto_action)
        else:
            max_forces = np.ones(self.robot.num_dofs) * 200
            max_forces[-2:] = 20
            self.robot.set_joint_position(
                tacto_action, max_forces, use_joint_effort_limits=False
            )

    def _get_obs(self):
        arm_joint_name_list = [
            "right_j0",
            "right_j1",
            "right_j2",
            "right_j3",
            "right_j4",
            "right_j5",
            "right_j6",
        ]
        gripper_joint_name_list = [
            "base_joint_gripper_left",
            "base_joint_gripper_right",
        ]

        arm_joint_idx = [
            self.robot.get_joint_index_by_name(joint_name)
            for joint_name in arm_joint_name_list
        ]
        arm_joint_state = self.robot.get_joint_states(joint_indices=arm_joint_idx)
        arm_joint_pos = arm_joint_state["joint_position"]
        arm_joint_vel = arm_joint_state["joint_velocity"]

        gripper_joint_idx = [
            self.robot.get_joint_index_by_name(joint_name)
            for joint_name in gripper_joint_name_list
        ]
        gripper_joint_state = self.robot.get_joint_states(
            joint_indices=gripper_joint_idx
        )
        gripper_joint_pos = gripper_joint_state["joint_position"][
            1:
        ]  # TODO: 絶対値を2で割る?
        gripper_joint_vel = np.zeros(1)

        eef_joint_state = self.robot.get_joint_state_by_name("right_hand")
        wrench = eef_joint_state["joint_reaction_forces"]

        return {
            "joint_pos": np.concatenate(
                (arm_joint_pos, gripper_joint_pos),
                dtype=np.float64,
            ),
            "joint_vel": np.concatenate(
                (arm_joint_vel, gripper_joint_vel),
                dtype=np.float64,
            ),
            "wrench": wrench,
        }

    def _get_info(self):
        info = {}

        if len(self.camera_names) + len(self.rgb_tactile_names) == 0:
            return info

        info["rgb_images"] = {}
        info["depth_images"] = {}

        for camera_name, camera in self.cameras.items():
            rgb_image, depth_image = camera.get_image()
            info["rgb_images"][camera_name] = rgb_image
            info["depth_images"][camera_name] = depth_image

        rgb_tactiles, depth_tactiles = self.rgb_tactiles.render()
        for tactile_name, rgb_tactile, depth_tactile in zip(
            self.rgb_tactile_names, rgb_tactiles, depth_tactiles
        ):
            info["rgb_images"][tactile_name] = rgb_tactile
            info["depth_images"][tactile_name] = depth_tactile

        return info

    def _get_reward(self):
        return 0.0

    def close(self):
        pass

    def get_joint_pos_from_obs(self, obs):
        """Get joint position from observation."""
        return obs["joint_pos"]

    def get_joint_vel_from_obs(self, obs):
        """Get joint velocity from observation."""
        return obs["joint_vel"]

    def get_gripper_joint_pos_from_obs(self, obs):
        """Get gripper joint position from observation."""
        joint_pos = self.get_joint_pos_from_obs(obs)
        gripper_joint_pos = np.zeros(
            DataKey.get_dim(DataKey.COMMAND_GRIPPER_JOINT_POS, self)
        )

        for body_config in self.body_config_list:
            if not isinstance(body_config, ArmConfig):
                continue

            gripper_joint_pos[body_config.gripper_joint_idxes_in_gripper_joint_pos] = (
                joint_pos[body_config.gripper_joint_idxes]
            )

        return gripper_joint_pos

    def get_eef_wrench_from_obs(self, obs):
        """Get end-effector wrench (fx, fy, fz, nx, ny, nz) from observation."""
        return obs["wrench"]

    def get_time(self):
        """Get simulation time. [s]"""
        return time.time() - self.init_time

    def get_link_pose(self, link_name):
        """Get link pose (tx, ty, tz, qw, qx, qy, qz)."""
        link_state = self.robot.get_link_state_by_name(link_name)
        return np.concatenate(
            (link_state.link_world_position, link_state.link_world_orientation)
        )

    @property
    def camera_names(self):
        """Get camera names."""
        return list(self.cameras.keys())

    @property
    def pointcloud_camera_names(self):
        """Get pointcloud camera names."""
        return []

    @property
    def rgb_tactile_names(self):
        """Get names of tactile sensors with RGB output."""
        return ["tactile_left", "tactile_right"]

    @property
    def intensity_tactile_names(self):
        """Get names of tactile sensors with intensity output."""
        return list(self.intensity_tactiles.keys())

    def get_camera_fovy(self, camera_name):
        """Get vertical field-of-view of the camera."""
        return self.cameras[camera_name].fov

    @abstractmethod
    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        pass

    def draw_box_marker(self, pos, mat, size, rgba):
        """Draw box marker."""
        pass

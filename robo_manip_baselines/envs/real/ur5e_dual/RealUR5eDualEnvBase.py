import time
from os import path

import numpy as np
import rtde_control
import rtde_receive
from gello.robots.robotiq_gripper import RobotiqGripper
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import ArmConfig
from robo_manip_baselines.teleop import (
    GelloInputDevice,
    KeyboardInputDevice,
    SpacemouseInputDevice,
)

from ..RealEnvBase import RealEnvBase


class RealUR5eDualEnvBase(RealEnvBase):
    action_space = Box(
        low=np.array(
            [
                -2 * np.pi,
                -2 * np.pi,
                -1 * np.pi,
                -2 * np.pi,
                -2 * np.pi,
                -2 * np.pi,
                0.0,
            ]
            * 2,
            dtype=np.float32,
        ),
        high=np.array(
            [
                2 * np.pi,
                2 * np.pi,
                1 * np.pi,
                2 * np.pi,
                2 * np.pi,
                2 * np.pi,
                255.0,
            ]
            * 2,
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64),
        }
    )

    def __init__(
        self,
        robot_ip_left,  # left arm from workspace side
        robot_ip_right,  # right arm from workspace side
        camera_ids,
        gelsight_ids,
        init_qpos,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Setup robot
        self.init_qpos = init_qpos
        self.joint_vel_limit = np.deg2rad(191)  # [rad/s]
        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__), "../../assets/common/robots/ur5e/ur5e.urdf"
                ),
                arm_root_pose=None,
                ik_eef_joint_id=6,
                arm_joint_idxes=np.arange(6),
                gripper_joint_idxes=np.array([6]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[0:6],
                init_gripper_joint_pos=np.zeros(1),
            ),
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__), "../../assets/common/robots/ur5e/ur5e.urdf"
                ),
                arm_root_pose=None,
                ik_eef_joint_id=6,
                arm_joint_idxes=np.arange(7, 13),
                gripper_joint_idxes=np.array([13]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([1]),
                eef_idx=1,
                init_arm_joint_pos=self.init_qpos[7:13],
                init_gripper_joint_pos=np.zeros(1),
            ),
        ]

        # Connect to UR5eDual
        print(f"[{self.__class__.__name__}] Start connecting the UR5eDual.")

        self.robot_ip_left = robot_ip_left
        self.rtde_c_left = rtde_control.RTDEControlInterface(self.robot_ip_left)
        self.rtde_r_left = rtde_receive.RTDEReceiveInterface(self.robot_ip_left)
        self.rtde_c_left.endFreedriveMode()

        self.robot_ip_right = robot_ip_right
        self.rtde_c_right = rtde_control.RTDEControlInterface(self.robot_ip_right)
        self.rtde_r_right = rtde_receive.RTDEReceiveInterface(self.robot_ip_right)
        self.rtde_c_right.endFreedriveMode()

        self.arm_joint_pos_actual = np.concatenate(
            [
                np.array(self.rtde_r_left.getActualQ()),
                np.array(self.rtde_r_right.getActualQ()),
            ]
        )
        print(f"[{self.__class__.__name__}] Finish connecting the UR5eDual.")

        # Connect to Robotiq gripper
        print(f"[{self.__class__.__name__}] Start connecting the Robotiq gripper.")
        self.gripper_port = 63352

        self.gripper_left = RobotiqGripper()
        self.gripper_left.connect(hostname=self.robot_ip_left, port=self.gripper_port)

        self.gripper_right = RobotiqGripper()
        self.gripper_right.connect(hostname=self.robot_ip_right, port=self.gripper_port)

        self._gripper_activated = False
        print(f"[{self.__class__.__name__}] Finish connecting the Robotiq gripper.")

        # Connect to RealSense
        self.setup_realsense(camera_ids)
        self.setup_gelsight(gelsight_ids)

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
                body_manager,
                **{
                    **default_kwargs.get(device_idx, {}),
                    **overwrite_kwargs.get(device_idx, {}),
                },
            )
            for device_idx, body_manager in enumerate(motion_manager.body_manager_list)
        ]

    def get_input_device_kwargs(self, input_device_name):
        return {}

    def _reset_robot(self):
        print(
            f"[{self.__class__.__name__}] Start moving the robot to the reset position."
        )
        self._set_action(
            self.init_qpos, duration=None, joint_vel_limit_scale=0.3, wait=True
        )
        print(
            f"[{self.__class__.__name__}] Finish moving the robot to the reset position."
        )

        if not self._gripper_activated:
            self._gripper_activated = True
            print(f"[{self.__class__.__name__}] Start activating the Robotiq gripper.")
            self.gripper_left.activate()
            self.gripper_right.activate()
            print(f"[{self.__class__.__name__}] Finish activating the Robotiq gripper.")

        # Calibrate force sensor
        time.sleep(0.2)
        self.rtde_c_left.zeroFtSensor()
        self.rtde_c_right.zeroFtSensor()
        time.sleep(0.2)

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        start_time = time.time()

        # Overwrite duration or joint_pos for safety
        action, duration = self.overwrite_command_for_safety(
            action, duration, joint_vel_limit_scale
        )

        # Send command to UR5eDual
        velocity = 0.5
        acceleration = 0.5
        lookahead_time = 0.2  # [s]
        gain = 100

        arm_joint_pos_command = action[self.body_config_list[0].arm_joint_idxes]
        period = self.rtde_c_left.initPeriod()
        self.rtde_c_left.servoJ(
            arm_joint_pos_command,
            velocity,
            acceleration,
            duration,
            lookahead_time,
            gain,
        )

        arm_joint_pos_command = action[self.body_config_list[1].arm_joint_idxes]
        period = self.rtde_c_right.initPeriod()
        self.rtde_c_right.servoJ(
            arm_joint_pos_command,
            velocity,
            acceleration,
            duration,
            lookahead_time,
            gain,
        )

        self.rtde_c_left.waitPeriod(period)
        self.rtde_c_right.waitPeriod(period)

        # Send command to Robotiq gripper
        speed = 50
        force = 10

        gripper_pos = action[self.body_config_list[0].gripper_joint_idxes][0]
        self.gripper_left.move(int(gripper_pos), speed, force)

        gripper_pos = action[self.body_config_list[1].gripper_joint_idxes][0]
        self.gripper_right.move(int(gripper_pos), speed, force)

        # Wait
        elapsed_duration = time.time() - start_time
        if wait and elapsed_duration < duration:
            time.sleep(duration - elapsed_duration)

    def _get_obs(self):
        # Get state from UR5eDual
        arm_joint_pos_left = np.array(self.rtde_r_left.getActualQ())
        arm_joint_vel_left = np.array(self.rtde_r_left.getActualQd())
        arm_joint_pos_right = np.array(self.rtde_r_right.getActualQ())
        arm_joint_vel_right = np.array(self.rtde_r_right.getActualQd())
        self.arm_joint_pos_actual = np.concatenate(
            [arm_joint_pos_left, arm_joint_pos_right], dtype=np.float64
        )

        # Get state from Robotiq gripper
        gripper_joint_pos_left = np.array(
            [self.gripper_left.get_current_position()], dtype=np.float64
        )
        gripper_joint_vel_left = np.zeros(1)
        gripper_joint_pos_right = np.array(
            [self.gripper_right.get_current_position()], dtype=np.float64
        )
        gripper_joint_vel_right = np.zeros(1)

        # Get wrench from force sensor
        wrench = np.concatenate(
            (
                self.rtde_r_left.getActualTCPForce(),
                self.rtde_r_right.getActualTCPForce(),
            ),
            dtype=np.float64,
        )

        return {
            "joint_pos": np.concatenate(
                (
                    arm_joint_pos_left,
                    gripper_joint_pos_left,
                    arm_joint_pos_right,
                    gripper_joint_pos_right,
                ),
                dtype=np.float64,
            ),
            "joint_vel": np.concatenate(
                (
                    arm_joint_vel_left,
                    gripper_joint_vel_left,
                    arm_joint_vel_right,
                    gripper_joint_vel_right,
                ),
                dtype=np.float64,
            ),
            "wrench": wrench,
        }

import numpy as np
from pyquaternion import Quaternion
from ee_sim_env import GOAL_POSE

import IPython
e = IPython.embed


class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        dt = (next_waypoint["t"] - curr_waypoint["t"])
        if dt == 0:
            t_frac = 0.0
        else:
            t_frac = (t - curr_waypoint["t"]) / dt
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts, type=None):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts, type)
            # initialize current waypoint
            if self.right_trajectory:
                self.curr_right_waypoint = self.right_trajectory[0]

        # if траектория закончилась — удерживаем последнюю позу
        if not self.right_trajectory:
            right_xyz = self.curr_right_waypoint['xyz']
            right_quat = self.curr_right_waypoint['quat']
            right_gripper = self.curr_right_waypoint['gripper']
        else:
            # obtain left and right waypoints
            if self.right_trajectory and self.right_trajectory[0]['t'] == self.step_count:
                self.curr_right_waypoint = self.right_trajectory.pop(0)
            next_right_waypoint = self.right_trajectory[0] if self.right_trajectory else self.curr_right_waypoint

            # interpolate between waypoints to obtain current pose and gripper command
            right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)


        # Inject noise
        if self.inject_noise:
            scale = 0.01
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_right])


class PickTransferCube(BasePolicy):

    def generate_trajectory(self, ts_first, type):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_xyz = np.array([0, 0.5, 0.25])

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, 
        ]

class PickTransferTorus(BasePolicy):

    def generate_trajectory(self, ts_first, type):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        # print(box_xyz)

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_xyz = np.array([0, 0.5, 0.25])
        
        # radius = 2.1 / 2
        # scale = 0.07
        # x_shift = 0 + radius * scale
        # y_shift = 0 + radius * scale
        # z_shift = -0.09
        x_shift = 0
        y_shift = -0.07
        z_shift = -0.1

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1}, # sleep
            {"t": 90, "xyz": box_xyz + np.array([x_shift, y_shift, 0.3]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([x_shift, y_shift, z_shift]), "quat": gripper_pick_quat.elements, "gripper": 1}, # go down
            {"t": 170, "xyz": box_xyz +  np.array([x_shift, y_shift, z_shift]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0}, # approach meet position
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # move to meet position
            {"t": 250, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0}, # stay
        ]

class PickTransferMixCube(BasePolicy):

    def generate_trajectory(self, ts_first, type):
        # print("GENERATING TRAJ FOR TYPE ", type)
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']

        box_info = np.array(ts_first.observation['env_state'])
        # Define cubes explicitly: box1 (green) is first 7 dof, box2 (red) is second 7 dof
        green_xyz = box_info[:3]
        red_xyz = box_info[7:10]
        goal_xyz = GOAL_POSE[0]  # [x, y, z]

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)
        
        # Heights / offsets
        lift_height = 0.15
        hover_above = 0.10
        cube_size = 0.04  # full edge length (approx)
        stack_height = cube_size  # distance between cube centers when stacked

        # Key target poses
        above_red = red_xyz + np.array([0, 0, lift_height])
        grasp_red = red_xyz + np.array([0, 0, -0.01])
        above_goal = goal_xyz + np.array([0, 0, hover_above])
        place_red = goal_xyz + np.array([0, 0, cube_size / 2])

        above_green = green_xyz + np.array([0, 0, lift_height])
        grasp_green = green_xyz + np.array([0, 0, -0.01])
        stack_above_goal = goal_xyz + np.array([0, 0, hover_above + stack_height])
        place_green = goal_xyz + np.array([0, 0, cube_size + cube_size / 2])

        retreat_high = np.array([goal_xyz[0], goal_xyz[1], goal_xyz[2] + lift_height + stack_height])
        start_mid = init_mocap_pose_right[:3] + np.array([0, 0, 0.15])

        # Trajectory: pick red -> place in goal -> pick green -> stack on red -> return
        # Timings stretched to ~6s (dt=0.02 -> ~300 steps total)
        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 1},
            {"t": 15, "xyz": above_red, "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 38, "xyz": grasp_red, "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 60, "xyz": grasp_red + np.array([0, 0, -0.005]), "quat": gripper_pick_quat.elements, "gripper": 0},  # close
            {"t": 83, "xyz": above_red, "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 105, "xyz": above_goal, "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 128, "xyz": place_red + np.array([0, 0, 0.01]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 150, "xyz": place_red, "quat": gripper_pick_quat.elements, "gripper": 1},  # release red
            {"t": 165, "xyz": above_goal, "quat": gripper_pick_quat.elements, "gripper": 1},

            {"t": 188, "xyz": above_green, "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 210, "xyz": grasp_green, "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 233, "xyz": grasp_green + np.array([0, 0, -0.005]), "quat": gripper_pick_quat.elements, "gripper": 0},  # close
            {"t": 255, "xyz": above_green, "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 270, "xyz": stack_above_goal, "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 283, "xyz": place_green + np.array([0, 0, 0.01]), "quat": gripper_pick_quat.elements, "gripper": 0},
            {"t": 292, "xyz": place_green, "quat": gripper_pick_quat.elements, "gripper": 1},  # release green
            {"t": 298, "xyz": retreat_high, "quat": gripper_pick_quat.elements, "gripper": 1},
            {"t": 300, "xyz": start_mid, "quat": gripper_pick_quat.elements, "gripper": 1},
        ]


if __name__ == '__main__':
    print("scripted_policy.py executed")

import numpy as np
from pyquaternion import Quaternion

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
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
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

        # obtain left and right waypoints
        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

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
        if not type: # Green cube (type=0)
            box_xyz = box_info[:3] # FIRST CUBE (green)
            target_zone = np.array([0.0, 0.7, 0.08])  # Green zone
        else:  # Red cube (type=1)
            box_xyz = box_info[7:10] # SECOND CUBE (red)
            target_zone = np.array([0.15, 0.7, 0.08])  # Red zone
        # print("CUBE1:", box_info[:3],"\nCUBE2:", box_info[7:10],"\ntraj_XYZ:", box_xyz)

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        # Retract position after placing cube
        retract_position = target_zone + np.array([0, 0, 0.25])

        self.right_trajectory = [
            # Phase 1: Approach cube
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # start position
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1}, # approach above cube

            # Phase 2: Grasp
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1}, # descend to cube
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0}, # close gripper (grasp!)

            # Phase 3: Lift cube
            {"t": 210, "xyz": box_xyz + np.array([0, 0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0}, # lift up

            # Phase 4: Move to target zone
            {"t": 250, "xyz": target_zone + np.array([0, 0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0}, # above target zone

            # Phase 5: Place in zone
            {"t": 290, "xyz": target_zone, "quat": gripper_pick_quat.elements, "gripper": 0}, # lower into zone
            {"t": 310, "xyz": target_zone, "quat": gripper_pick_quat.elements, "gripper": 1}, # open gripper (release!)

            # Phase 6: Retract gripper upward
            {"t": 350, "xyz": retract_position, "quat": gripper_pick_quat.elements, "gripper": 1}, # lift gripper up
            {"t": 400, "xyz": retract_position, "quat": gripper_pick_quat.elements, "gripper": 1}, # hold position
        ]


if __name__ == '__main__':
    print("scripted_policy.py executed")

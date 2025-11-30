import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
import multiprocessing as mp

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env, set_goal_zone_pose as set_goal_zone_pose_ee
from sim_env import make_sim_env, BOX_POSE, set_goal_zone_pose as set_goal_zone_pose_sim
from scripted_policy import PickTransferCube, PickTransferTorus, PickTransferMixCube
from utils import sample_goal_zone_pose

import IPython
e = IPython.embed


def generate_episodes(task_name, dataset_dir, num_episodes, onscreen_render, start_idx=0):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    inject_noise = False
    render_cam_name = 'angle'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name in {'multiple_red', 'multiple_blue', 'multiple_green', 'single_cube'}:
        policy_cls = PickTransferCube
    elif task_name in {"single_torus"}:
        policy_cls = PickTransferTorus
    elif task_name in {'mix_cube'}:
        policy_cls = PickTransferMixCube
    else:
        raise NotImplementedError

    success = []
    episode_idx = start_idx - 1
    last_idx = start_idx + num_episodes - 1
    while episode_idx < last_idx:
        episode_idx += 1
        print(f'worker={os.getpid()} episode_idx={episode_idx}')
        if task_name == 'mix_cube':
            goal_pose = sample_goal_zone_pose()
            set_goal_zone_pose_ee(goal_pose)
            set_goal_zone_pose_sim(goal_pose)
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise)
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for step in range(episode_len):
            action = policy(ts, episode_idx % 2)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.002)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed -> RESTARTING ATTEMPT")
            episode_idx = episode_idx - 1
            continue

        joint_traj = [ts.observation['qpos'] for ts in episode]
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            joint[6] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy() # box poses at step 0

        # clear unused variables
        del env
        del episode
        del policy

        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()

        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.02)

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed -> RESTARTING ATTEMPT")
            episode_idx = episode_idx - 1
            continue

        plt.close()

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
            '/observations/type': [],
        }

        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            
            if 'mix_cube' == task_name:
                data_dict['/observations/type'].append(episode_idx%2)
            else:
                data_dict['/observations/type'].append(None)

        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 7))
            qvel = obs.create_dataset('qvel', (max_timesteps, 7))
            action = root.create_dataset('action', (max_timesteps, 7))

            TYPE = obs.create_dataset('type', (max_timesteps))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--num_workers', action='store', type=int, default=1, help='parallel workers for generation')
    parser.add_argument('--start_index', action='store', type=int, default=0, help='starting episode index (e.g., 41 to continue)')
    
    args = vars(parser.parse_args())
    num_workers = max(1, args.get('num_workers', 1))
    start_index = max(0, args.get('start_index', 0))

    if num_workers == 1:
        generate_episodes(args['task_name'], args['dataset_dir'], args['num_episodes'], args['onscreen_render'], start_idx=start_index)
    else:
        ctx = mp.get_context("spawn")
        per_worker = args['num_episodes'] // num_workers
        remainder = args['num_episodes'] % num_workers
        procs = []
        start = start_index
        for wi in range(num_workers):
            count = per_worker + (1 if wi < remainder else 0)
            if count == 0:
                continue
            p = ctx.Process(target=generate_episodes, args=(args['task_name'], args['dataset_dir'], count, False, start))
            p.start()
            procs.append(p)
            start += count
        for p in procs:
            p.join()

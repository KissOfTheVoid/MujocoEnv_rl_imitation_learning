import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from einops import rearrange

from constants import DT
from utils import sample_box_pose # robot functions
from utils import set_seed  # helper functions
from policy import ACTPolicy
import cv2

from sim_env import BOX_POSE

import IPython

import csv
e = IPython.embed

def main(args):
    # with open("zz_final_aloha/ordered_products.csv") as file:
    #     orders
    orders = []
    with open(args['csv_path']) as csvfile:
        file_reader = csv.reader(csvfile, delimiter=",")
        for idx, row in enumerate(file_reader):
            if idx == 0:
                continue
            orders.append(torch.tensor(int(row[1])))
    print(orders)
    # exit()
    set_seed(1)
    # command line parameters
    policy_class = 'ACT'
    onscreen_render = True
    task_name = 'prototype'

    # get task parameters
    from constants import SIM_TASK_CONFIGS
    task_config = SIM_TASK_CONFIGS[task_name]
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # # fixed parameters
    state_dim = 7
    if policy_class == 'ACT':
        policy_config = {'lr': 1e-5,
                         'num_queries': 100,
                         'kl_weight': 10,
                         'hidden_dim': 512,
                         'dim_feedforward': 3200,
                         'lr_backbone': 1e-5,
                         'backbone': 'resnet18',
                         'enc_layers': 4,
                         'dec_layers': 7,
                         'nheads': 8,
                         'camera_names': camera_names,
                         }

    config = {
        'num_epochs': 1000,
        'ckpt_dir': args['ckpt_dir'],
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': 1e-5,
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': 0,
        'temporal_agg': True,
        'camera_names': camera_names,
    }

    ckpt_name = [f'policy_best.ckpt']
    # eval_bc(config, ckpt_name, ckpt_dir=config['ckpt_dir'], orders=orders, save_episode=True)

    set_seed(2)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = 'prototype'
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    from sim_env import make_sim_env
    env = make_sim_env(task_name)
    env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    ### set task
    BOX_POSE[0] = np.concatenate((sample_box_pose(),sample_box_pose())) # used in sim reset
    ts = env.reset()

    ### onscreen render
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
        plt.ion()

    image_list = [] # for visualization
    policy_names = "policy_best.ckpt"
    policy_paths = config['ckpt_dir']


    policy, pre_process, post_process = initialize_policy(ckpt_name=policy_names, 
                                                              ckpt_dir=policy_paths, 
                                                              policy_class=policy_class, 
                                                              policy_config=policy_config)
    for o in range(len(orders)):
        # policy, pre_process, post_process = initialize_policy(ckpt_name=policy_names, 
        #                                                       ckpt_dir=policy_paths, 
        #                                                       policy_class=policy_class, 
        #                                                       policy_config=policy_config)
        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim])

        qpos_history = torch.zeros((1, max_timesteps, state_dim))
        qpos_list = []
        target_qpos_list = []
        rewards = []
        
        # GRASPING OBJECT
        print("GRASPING OBJECT W/ type", orders[o])
        with torch.inference_mode():
            for t in range(200):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ### query policy
                # if config['policy_class'] == "ACT":
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image, type=orders[o])
                if temporal_agg:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]
                # else:
                #     raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action
                ### step the environment
                ts = env.step(target_qpos) 
                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
            # Open gripper and repeat for each item
            for t in range(10):
                cur = qpos_list[-1]
                cur[-1] = 1.1 # open_gripper
                ts = env.step(cur)
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
            for t in range(1):
                ts = env.step(qpos_list[0])

    plt.close()

    # if save_episode:
    save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{1}.mp4'))
    return 0

def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().unsqueeze(0)
    return curr_image

# load policy and stats
def initialize_policy(ckpt_name, ckpt_dir, policy_class, policy_config):

    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    ckpt_path = os.path.join(str(ckpt_dir), str(ckpt_name))
    policy = ACTPolicy(policy_config)
    # Device selection with MPS support
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device for loading")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for loading")
    else:
        device = torch.device("cpu")
        print("Using CPU device for loading")
    loading_status = policy.load_state_dict(
        torch.load(ckpt_path, map_location=device)['policy_state_dict']
    )
    print(loading_status)

    policy.to(torch.device("cpu"))
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    return policy, pre_process, post_process

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', action='store', type=str, help='dir with model', required=True)
    parser.add_argument('--csv_path', action='store', type=str, help='user order request csv', required=True)

    main(vars(parser.parse_args()))
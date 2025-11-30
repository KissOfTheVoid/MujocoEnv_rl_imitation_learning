"""
Запуск одной симуляции по сцене из JSON.
Используется сценой редактора (scene_editor.py) как внешняя утилита.
"""
import sys
import os
import json
from pathlib import Path

import numpy as np
import torch
import pickle
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP_DIR = ROOT / "experiments"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

# workspace bounds (meters) aligned with mix_cube dataset
WORK_X = (0.0, 0.30)
WORK_Y = (0.35, 0.85)


def log(msg):
    print(msg, flush=True)


def run_simulation(scene_data):
    from experiments.sim_env import make_sim_env, BOX_POSE, set_goal_zone_pose
    from experiments.constants import SIM_TASK_CONFIGS, DT
    from experiments.policy import ACTPolicy
    from experiments.imitate_episodes import get_image
    from experiments.utils import set_seed

    ckpt_dir = ROOT / "experiments" / "checkpoints" / "mix_cube"
    stats_path = ckpt_dir / "dataset_stats.pkl"
    ckpt_path = ckpt_dir / "policy_best.ckpt"

    cubes = [o for o in scene_data if o["kind"] == "cube"]
    goals = [o for o in scene_data if o["kind"] == "goal"]
    if len(cubes) < 2 or len(goals) < 1:
        log("Нужно минимум два куба и один goal_zone.")
        return 1

    set_seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = SIM_TASK_CONFIGS["mix_cube"]
    camera_names = cfg["camera_names"]
    episode_len = cfg["episode_len"]

    def clamp_pose(d):
        x, y, z = d["pos"]
        x = float(np.clip(x, WORK_X[0], WORK_X[1]))
        y = float(np.clip(y, WORK_Y[0], WORK_Y[1]))
        z = float(np.clip(z, 0.0, 0.1))
        quat = np.array([1, 0, 0, 0])
        return np.array([x, y, z, *quat])

    BOX_POSE[0] = np.concatenate((clamp_pose(cubes[0]), clamp_pose(cubes[1])))
    goal = goals[0]
    gx = float(np.clip(goal["pos"][0], WORK_X[0], WORK_X[1]))
    gy = float(np.clip(goal["pos"][1], WORK_Y[0], WORK_Y[1]))
    set_goal_zone_pose(np.array([gx, gy, 0.001]))

    # apply colors (RGBA 0-255 -> 0-1) for two cubes
    cube_colors = []
    for c in cubes[:2]:
        r, g, b, a = c.get("color", (0, 255, 0, 255))
        cube_colors.append([r / 255.0, g / 255.0, b / 255.0, a / 255.0])

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
    post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

    policy_config = {
        "lr": 1e-5,
        "num_queries": 100,
        "kl_weight": 10,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": camera_names,
    }
    # ensure detr.main argparse has required fields
    argv_backup = sys.argv[:]
    sys.argv = [
        "scene_worker",
        "--ckpt_dir", str(ckpt_dir),
        "--policy_class", "ACT",
        "--task_name", "mix_cube",
        "--seed", "0",
        "--num_epochs", "1",
    ]
    policy = ACTPolicy(policy_config)
    sys.argv = argv_backup
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["policy_state_dict"] if "policy_state_dict" in ckpt else ckpt
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()

    env = make_sim_env("mix_cube")
    ts = env.reset()
    # set geom colors in the physics model (names: red_box, red_box2)
    try:
        physics = env._physics
        if len(cube_colors) >= 1:
            physics.named.model.geom_rgba["red_box"] = cube_colors[0]
        if len(cube_colors) >= 2:
            physics.named.model.geom_rgba["red_box2"] = cube_colors[1]
    except Exception:
        pass
    temporal_agg = True
    num_queries = policy_config["num_queries"]
    query_frequency = 1
    state_dim = 7
    max_timesteps = episode_len
    onscreen_cam = "angle"

    all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim], device=device)
    qpos_history = torch.zeros((1, max_timesteps, state_dim), device=device)
    image_list = []
    log(f"Running rollout on {device}...")
    with torch.inference_mode():
        for t in range(max_timesteps):
            obs = ts.observation
            image_list.append(obs["images"])
            qpos_numpy = np.array(obs["qpos"])
            qpos = pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().unsqueeze(0).to(device)
            qpos_history[:, t] = qpos
            curr_image = get_image(ts, camera_names).to(device)

            if t % query_frequency == 0:
                obj_type = torch.tensor(0, device=device, dtype=torch.long)
                all_actions = policy(qpos, curr_image, type=obj_type)
            all_time_actions[[t], t : t + num_queries] = all_actions
            actions_for_curr_step = all_time_actions[:, t]
            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)
            ts = env.step(action)

    h, w, _ = image_list[0][camera_names[0]].shape
    fps = int(1 / DT)
    out_path = ROOT / "scene_editor_sim.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in image_list:
        img = frame[onscreen_cam][:, :, [2, 1, 0]]  # BGR
        writer.write(img)
    writer.release()
    log(f"Done. Saved video to {out_path}")
    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="path to scene json")
    args = parser.parse_args()
    with open(args.scene, "r") as f:
        data = json.load(f)
    code = run_simulation(data)
    sys.exit(code)


if __name__ == "__main__":
    main()

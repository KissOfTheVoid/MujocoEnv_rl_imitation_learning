"""
3D редактор сцены на PySide6 с рендером MuJoCo (offscreen Renderer).
Клики по сцене выделяют объекты (mjv_select), перемещение — перетаскиванием мышью в режиме Move
или кнопками-стрелками. Интерфейс в стиле Unity с иконками.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import mujoco as mj
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
import pickle
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXP_DIR = ROOT / "experiments"
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

# Рабочая область MuJoCo (метры)
WORK_X = (0.0, 0.30)
WORK_Y = (0.35, 0.85)
WORK_Z = (0.0, 0.10)
DEFAULT_CUBE_Z = 0.05
DEFAULT_GOAL_Z = 0.001
BASE_XML = ROOT / "experiments" / "assets" / "mix_cube" / "bimanual_viperx_ee_transfer_cube.xml"

_id_counter = 0


def _new_id():
    global _id_counter
    _id_counter += 1
    return _id_counter


def _clamp_workspace(pos: np.ndarray) -> np.ndarray:
    pos[0] = float(np.clip(pos[0], WORK_X[0], WORK_X[1]))
    pos[1] = float(np.clip(pos[1], WORK_Y[0], WORK_Y[1]))
    pos[2] = float(np.clip(pos[2], WORK_Z[0], WORK_Z[1]))
    return pos


def _geom_id(model: mj.MjModel, name: str) -> int:
    try:
        return mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
    except Exception:
        return -1


def _body_id(model: mj.MjModel, name: str) -> int:
    try:
        return mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
    except Exception:
        return -1


def _euler_to_wxyz(rot_deg: np.ndarray) -> np.ndarray:
    """MuJoCo ожидает кватернион в формате w, x, y, z."""
    quat_xyzw = R.from_euler("xyz", rot_deg, degrees=True).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


@dataclass
class SceneObject:
    kind: str  # "cube" или "goal"
    name: str
    pos: np.ndarray
    rot: np.ndarray  # Euler deg
    color: QtGui.QColor = field(default_factory=lambda: QtGui.QColor.fromHsv(np.random.randint(0, 359), 190, 230))
    geom_name: str = ""
    body_name: str = ""
    geom_names: List[str] = field(default_factory=list)


class MujocoViewport(QtWidgets.QWidget):
    picked = QtCore.Signal(object)  # SceneObject or None
    request_refresh = QtCore.Signal()

    def __init__(self, model: Dict[int, SceneObject]):
        super().__init__()
        self.setMouseTracking(True)
        self.model_dict = model
        self.objects: List[SceneObject] = []
        self.mj_model: Optional[mj.MjModel] = None
        self.mj_data: Optional[mj.MjData] = None
        self.renderer: Optional[mj.Renderer] = None
        self.cam = mj.MjvCamera()
        mj.mjv_defaultCamera(self.cam)
        self.cam.distance = 1.35
        self.cam.azimuth = -135
        self.cam.elevation = 30
        self.cam.lookat = np.array([0.15, 0.6, 0.04])
        self.vopt = mj.MjvOption()
        mj.mjv_defaultOption(self.vopt)
        self.image_label = QtWidgets.QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.image_label)

        self.geom_map: Dict[str, SceneObject] = {}
        self.body_map: Dict[str, SceneObject] = {}
        self.selected: Optional[SceneObject] = None
        self.drag_move = False
        self.last_hit: Optional[np.ndarray] = None
        self.last_mouse = QtCore.QPointF(0, 0)
        self.camera_drag = None
        self.keys_down: Set[int] = set()
        self.mode_getter = lambda: "select"
        self.sim_running = False
        self._backup_qpos = None
        self._backup_qvel = None
        self.sim_policy_running = False
        self.policy_env = None
        self.policy_ts = None
        self.policy_device = None
        self.policy_cfg = None
        self.policy_all_actions = None
        self.policy_qpos_history = None
        self.policy_num_queries = 0
        self.policy_query_frequency = 1
        self.policy_max_timesteps = 0
        self.policy_t = 0
        self.policy_policy = None
        self.policy_pre = None
        self.policy_post = None
        self.policy_camera_names = []

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(30)

        self._rebuild_model()

    def _goal_templates(self) -> List[ET.Element]:
        try:
            scene_path = BASE_XML.parent / "scene.xml"
            scene_tree = ET.parse(str(scene_path))
            scene_root = scene_tree.getroot()
            for b in scene_root.findall(".//body"):
                if b.get("name") == "goal_zone":
                    return [g for g in b if g.tag == "geom"]
        except Exception:
            pass
        return []

    # --- Model build -----------------------------------------------------
    def _build_xml(self) -> str:
        # Load base MuJoCo model (robot + table + goal zone), remove default cubes
        tree = ET.parse(str(BASE_XML))
        root = tree.getroot()
        vis = root.find("visual")
        if vis is None:
            vis = ET.SubElement(root, "visual")
        glob = vis.find("global")
        if glob is None:
            glob = ET.SubElement(vis, "global")
        glob.set("offwidth", "2048")
        glob.set("offheight", "2048")

        world = root.find("worldbody")
        if world is None:
            return ET.tostring(root, encoding="unicode")

        # remove default cubes and goal_zone from file
        for b in list(world):
            if b.tag == "body" and b.get("name") in {"box", "box2", "goal_zone"}:
                world.remove(b)

        # remove any keyframes/keys (base files may contain qpos sized for removed bodies)
        for parent in list(root.iter()):
            for child in list(parent):
                if child.tag == "keyframe" or child.tag == "key":
                    parent.remove(child)

        goal_geoms_template = self._goal_templates()

        # add goals from editor
        for obj in self.objects:
            if obj.kind != "goal":
                continue
            if not obj.body_name:
                obj.body_name = f"goal_{_new_id()}"
            quat = _euler_to_wxyz(obj.rot)
            rgba = obj.color.getRgbF()
            body = ET.SubElement(
                world,
                "body",
                name=obj.body_name,
                pos=f"{obj.pos[0]} {obj.pos[1]} {obj.pos[2]}",
                quat=f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
            )
            obj.geom_names = []
            for i, g in enumerate(goal_geoms_template or []):
                gg = ET.fromstring(ET.tostring(g))
                name = g.get("name", f"goal_geom_{i}")
                new_name = f"{obj.body_name}_{name}"
                gg.set("name", new_name)
                gg.set("rgba", f"{rgba[0]} {rgba[1]} {rgba[2]} 1")
                body.append(gg)
                obj.geom_names.append(new_name)
            # fallback simple plate if template missing
            if not obj.geom_names:
                gname = f"{obj.body_name}_geom"
                ET.SubElement(
                    body,
                    "geom",
                    name=gname,
                    type="box",
                    size="0.05 0.05 0.002",
                    rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} 0.5",
                    contype="0",
                    conaffinity="0",
                )
                obj.geom_names.append(gname)

        # add cubes from editor
        for obj in self.objects:
            if obj.kind != "cube":
                continue
            if not obj.body_name:
                obj.body_name = f"cube_{_new_id()}"
            if not obj.geom_name:
                obj.geom_name = f"{obj.body_name}_geom"
            obj.geom_names = [obj.geom_name]
            quat = _euler_to_wxyz(obj.rot)
            rgba = obj.color.getRgbF()
            body = ET.SubElement(world, "body", name=obj.body_name, pos=f"{obj.pos[0]} {obj.pos[1]} {obj.pos[2]}", quat=f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}")
            ET.SubElement(body, "joint", name=f"{obj.body_name}_joint", type="free", frictionloss="0.01")
            ET.SubElement(body, "inertial", pos="0 0 0", mass="0.05", diaginertia="0.002 0.002 0.002")
            ET.SubElement(
                body,
                "geom",
                name=obj.geom_name,
                type="box",
                size="0.02 0.02 0.02",
                pos="0 0 0",
                condim="4",
                solimp="2 1 0.01",
                solref="0.01 1",
                friction="1 0.005 0.0001",
                rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}",
            )

        return ET.tostring(root, encoding="unicode")

    def _rebuild_model(self):
        self.objects = list(self.model_dict.values())
        xml = self._build_xml()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xml", dir=BASE_XML.parent)
        tmp.write(xml.encode("utf-8"))
        tmp.close()
        try:
            self.mj_model = mj.MjModel.from_xml_path(tmp.name)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
        self.mj_data = mj.MjData(self.mj_model)
        # apply object poses to data qpos
        for obj in self.objects:
            self._update_obj_in_model(obj)
        mj.mj_forward(self.mj_model, self.mj_data)
        self.renderer = mj.Renderer(self.mj_model, width=max(10, self.width()), height=max(10, self.height()))
        self.geom_map = {}
        self.body_map = {}
        for obj in self.objects:
            if obj.geom_names:
                for gn in obj.geom_names:
                    self.geom_map[gn] = obj
            if obj.geom_name:
                self.geom_map[obj.geom_name] = obj
            if obj.body_name:
                self.body_map[obj.body_name] = obj
        self.selected = None
        self._render(force=True)

    def _policy_step_once(self):
        if not self.sim_policy_running or self.policy_env is None or self.policy_ts is None:
            return
        try:
            t = self.policy_t
            if t >= self.policy_max_timesteps:
                self.policy_ts = self.policy_env.reset()
                self.policy_t = 0
            obs = self.policy_ts.observation
            qpos_numpy = np.array(obs["qpos"])
            qpos = self.policy_pre(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().unsqueeze(0).to(self.policy_device)
            from experiments.imitate_episodes import get_image
            curr_image = get_image(self.policy_ts, self.policy_camera_names).to(self.policy_device)

            obj_type = torch.tensor(0, device=self.policy_device, dtype=torch.long)
            all_actions = self.policy_policy(qpos, curr_image, type=obj_type)
            raw_action = all_actions[0].detach().cpu().numpy()
            action = self.policy_post(raw_action)
            self.policy_ts = self.policy_env.step(action)
<<<<<<< ours
            # if episode done, reset to continue indefinitely
            try:
                if hasattr(self.policy_ts, "last") and self.policy_ts.last():
                    self.policy_ts = self.policy_env.reset()
            except Exception:
                pass

            # render fit to widget
            w = max(320, self.image_label.width())
            h = max(240, self.image_label.height())
            frame = self.policy_env.physics.render(height=h, width=w, camera_id="angle")
=======
            frame = self.policy_env.physics.render(height=480, width=640, camera_id="angle")
>>>>>>> theirs
            frame = np.ascontiguousarray(frame)  # dm_control returns RGB
            h, w, _ = frame.shape
            qimg = QtGui.QImage(frame.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
            self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg))
            self.policy_t += 1
        except Exception as e:
            print("Policy step error:", e)
            self.stop_policy_sim()

    # --- Selection & updates --------------------------------------------
    def _highlight(self):
        if not self.mj_model:
            return
        for obj in self.objects:
            geoms = obj.geom_names if obj.geom_names else [obj.geom_name]
            for gname in geoms:
                gid = _geom_id(self.mj_model, gname)
                if gid < 0:
                    continue
                rgba = np.array(obj.color.getRgbF(), dtype=float)
                if obj is self.selected:
                    rgba[:3] = np.clip(rgba[:3] + 0.25, 0, 1)
                    rgba[3] = 1.0
                self.mj_model.geom_rgba[gid] = rgba

    def _update_obj_in_model(self, obj: SceneObject):
        if not self.mj_model or not self.mj_data:
            return
        bid = _body_id(self.mj_model, obj.body_name)
        if bid < 0:
            return
        quat = _euler_to_wxyz(obj.rot)
        self.mj_model.body_pos[bid] = obj.pos
        self.mj_model.body_quat[bid] = quat
        # sync qpos for free joints so render updates immediately
        jname = f"{obj.body_name}_joint"
        jid = -1
        try:
            jid = mj.mj_name2id(self.mj_model, mj.mjtObj.mjOBJ_JOINT, jname)
        except Exception:
            jid = -1
        if jid >= 0:
            adr = self.mj_model.jnt_qposadr[jid]
            if adr >= 0 and adr + 7 <= self.mj_model.nq:
                self.mj_data.qpos[adr:adr+3] = obj.pos
                self.mj_data.qpos[adr+3:adr+7] = quat
        rgba = np.array(obj.color.getRgbF(), dtype=float)
        geoms = obj.geom_names if obj.geom_names else [obj.geom_name]
        for gname in geoms:
            gid = _geom_id(self.mj_model, gname)
            if gid >= 0:
                self.mj_model.geom_rgba[gid] = rgba
        mj.mj_forward(self.mj_model, self.mj_data)

    def set_selected(self, obj: Optional[SceneObject]):
        self.selected = obj
        self._highlight()

    def nudge(self, axis: int, sign: float, step: float):
        if not self.selected:
            return
        self.selected.pos[axis] += sign * step
        _clamp_workspace(self.selected.pos)
        self._update_obj_in_model(self.selected)
        self._render(force=True)

    def nudge_rot(self, axis: int, sign: float, step: float):
        if not self.selected:
            return
        self.selected.rot[axis] = (self.selected.rot[axis] + sign * step) % 360
        self._update_obj_in_model(self.selected)
        self._render(force=True)

    # --- Rendering -------------------------------------------------------
    def _render(self, force=False):
        if not self.renderer or not self.mj_data:
            return
        w, h = max(1, self.width()), max(1, self.height())
        max_w = getattr(self.mj_model.vis.global_, "offwidth", w)
        max_h = getattr(self.mj_model.vis.global_, "offheight", h)
        rw = int(min(max_w, w))
        rh = int(min(max_h, h))
        if force or (self.renderer.height != rh or self.renderer.width != rw):
            self.renderer = mj.Renderer(self.mj_model, rw, rh)
        self._highlight()
        self.renderer.update_scene(self.mj_data, self.cam, self.vopt)
        img = self.renderer.render()
        img = np.ascontiguousarray(img)  # OpenGL -> Qt, keep orientation as-is
        h, w, _ = img.shape
        qimg = QtGui.QImage(img.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _step_sim(self, dt=0.016):
        if not self.sim_running or not self.mj_model or not self.mj_data:
            return
        steps = max(1, int(dt / self.mj_model.opt.timestep))
        for _ in range(steps):
            mj.mj_step(self.mj_model, self.mj_data)

    def _on_timer(self):
        if self.sim_policy_running:
            self._policy_step_once()
            return
        elif self.sim_running:
            self._step_sim(0.03)
            self._render(force=True)
        else:
            self._render()

    def start_sim(self):
        if self.sim_running or not self.mj_data:
            return
        self._backup_qpos = self.mj_data.qpos.copy()
        self._backup_qvel = self.mj_data.qvel.copy()
        self.sim_running = True

    def stop_sim(self):
        if not self.sim_running or not self.mj_data:
            return
        if self._backup_qpos is not None and len(self._backup_qpos) == len(self.mj_data.qpos):
            self.mj_data.qpos[:] = self._backup_qpos
        if self._backup_qvel is not None and len(self._backup_qvel) == len(self.mj_data.qvel):
            self.mj_data.qvel[:] = self._backup_qvel
        mj.mj_forward(self.mj_model, self.mj_data)
        self.sim_running = False
        self._render(force=True)

    def start_policy_sim(self, objects: List[SceneObject]):
        if self.sim_policy_running:
            return
        from experiments.sim_env import make_sim_env, BOX_POSE, set_goal_zone_pose
        from experiments.constants import SIM_TASK_CONFIGS, DT
        from experiments.policy import ACTPolicy
        from experiments.imitate_episodes import get_image
        from experiments.utils import set_seed

        cubes = [o for o in objects if o.kind == "cube"]
        goals = [o for o in objects if o.kind == "goal"]

        def clamp_pose_obj(d: SceneObject):
            x, y, z = d.pos
            x = float(np.clip(x, WORK_X[0], WORK_X[1]))
            y = float(np.clip(y, WORK_Y[0], WORK_Y[1]))
            z = float(np.clip(z, 0.0, 0.1))
            quat = np.array([1, 0, 0, 0])
            return np.array([x, y, z, *quat])

        if len(cubes) >= 2:
            BOX_POSE[0] = np.concatenate((clamp_pose_obj(cubes[0]), clamp_pose_obj(cubes[1])))
        elif len(cubes) == 1:
            zero_pose = clamp_pose_obj(SceneObject("cube", "", np.array([0.05, 0.5, 0.05]), np.zeros(3)))
            BOX_POSE[0] = np.concatenate((clamp_pose_obj(cubes[0]), zero_pose))
        else:
            zero_pose = clamp_pose_obj(SceneObject("cube", "", np.array([0.05, 0.5, 0.05]), np.zeros(3)))
            BOX_POSE[0] = np.concatenate((zero_pose, zero_pose))

        if goals:
            g = goals[0]
            gx = float(np.clip(g.pos[0], WORK_X[0], WORK_X[1]))
            gy = float(np.clip(g.pos[1], WORK_Y[0], WORK_Y[1]))
            set_goal_zone_pose(np.array([gx, gy, g.pos[2]]))

        set_seed(0)
        ckpt_dir = ROOT / "experiments" / "checkpoints" / "mix_cube"
        stats_path = ckpt_dir / "dataset_stats.pkl"
        ckpt_path = ckpt_dir / "policy_best.ckpt"
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        pre_process = lambda s_qpos: (s_qpos - stats["qpos_mean"]) / stats["qpos_std"]
        post_process = lambda a: a * stats["action_std"] + stats["action_mean"]

        cfg = SIM_TASK_CONFIGS["mix_cube"]
        camera_names = cfg["camera_names"]
        episode_len = 10_000  # effectively no horizon for editor play
        device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

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
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["policy_state_dict"] if "policy_state_dict" in ckpt else ckpt
        policy.load_state_dict(state)
        policy.to(device)
        policy.eval()

        env = make_sim_env("mix_cube")
        ts = env.reset()

<<<<<<< ours
        # apply colors to env geoms if available
        try:
            cube_colors = []
            for c in cubes[:2]:
                r, g, b, a = c.color.getRgb()
                cube_colors.append([r / 255.0, g / 255.0, b / 255.0, a / 255.0])
            physics = env._physics
            if len(cube_colors) >= 1:
                physics.named.model.geom_rgba["red_box"] = cube_colors[0]
            if len(cube_colors) >= 2:
                physics.named.model.geom_rgba["red_box2"] = cube_colors[1]
        except Exception:
            pass

        # store (no temporal aggregation)
=======
        num_queries = policy_config["num_queries"]
        query_frequency = 1
        state_dim = 7
        all_time_actions = torch.zeros([episode_len, episode_len + num_queries, state_dim], device=device)
        qpos_history = torch.zeros((1, episode_len, state_dim), device=device)

        # store
>>>>>>> theirs
        self.policy_env = env
        self.policy_ts = ts
        self.policy_device = device
        self.policy_cfg = cfg
        self.policy_policy = policy
        self.policy_pre = pre_process
        self.policy_post = post_process
        self.policy_camera_names = camera_names
        self.policy_max_timesteps = episode_len
        self.policy_t = 0
        self.sim_policy_running = True

    def stop_policy_sim(self):
        self.sim_policy_running = False
        self.policy_env = None
        self.policy_ts = None
        self.policy_policy = None
        self.policy_all_actions = None
        self.policy_qpos_history = None
        self.policy_frame = None
        self.policy_t = 0
        self._render(force=True)

    # --- Mouse & keys ----------------------------------------------------
    def _relpos(self, event) -> Tuple[float, float]:
        w = max(1, self.width())
        h = max(1, self.height())
        x = event.position().x() / w
        y = 1.0 - event.position().y() / h
        return x, y

    def _select_at(self, event) -> Tuple[Optional[SceneObject], Optional[np.ndarray]]:
        if not self.mj_model or not self.renderer:
            return None, None
        relx, rely = self._relpos(event)
        selpnt = np.zeros(3, dtype=np.float64)
        geomid = np.array([-1], dtype=np.int32)
        skinid = np.array([-1], dtype=np.int32)
        aspect = self.width() / max(1, self.height())
        bodyid = mj.mjv_select(
            self.mj_model,
            self.mj_data,
            self.vopt,
            aspect,
            relx,
            rely,
            self.renderer.scene,
            selpnt,
            geomid,
            skinid,
        )
        obj = None
        if geomid[0] != -1:
            gname = mj.mj_id2name(self.mj_model, mj.mjtObj.mjOBJ_GEOM, geomid[0])
            obj = self.geom_map.get(gname)
        return obj, selpnt.copy()

    def mousePressEvent(self, event):
        self.last_mouse = event.position()
        mode = self.mode_getter()
        if event.button() == QtCore.Qt.LeftButton:
            if mode == "camera":
                self.camera_drag = "rotate"
            elif mode == "move":
                obj, hit = self._select_at(event)
                if obj:
                    self.drag_move = True
                    if obj is not self.selected:
                        self.set_selected(obj)
                        self.picked.emit(obj)
                    self.last_hit = hit
            else:
                obj, hit = self._select_at(event)
                if obj:
                    self.set_selected(obj)
                    self.picked.emit(obj)
                    self.last_hit = hit
        elif event.button() == QtCore.Qt.RightButton:
            self.camera_drag = "pan"

    def mouseMoveEvent(self, event):
        if self.camera_drag:
            delta = event.position() - self.last_mouse
            self.last_mouse = event.position()
            if self.camera_drag == "rotate":
                self.cam.azimuth -= delta.x() * 0.4
                self.cam.elevation = float(np.clip(self.cam.elevation - delta.y() * 0.3, -89, 89))
            elif self.camera_drag == "pan":
                factor = self.cam.distance * 0.0015
                right = np.array(
                    [math.cos(math.radians(self.cam.azimuth)), -math.sin(math.radians(self.cam.azimuth)), 0.0]
                )
                up = np.array([0, 0, 1.0])
                move = (-right * delta.x() + up * delta.y()) * factor
                self.cam.lookat = self.cam.lookat + move
            return
        if self.drag_move and self.selected:
            obj, hit = self._select_at(event)
            if hit is not None:
                new_pos = self.selected.pos.copy()
                new_pos[0], new_pos[1] = hit[0], hit[1]
                _clamp_workspace(new_pos)
                self.selected.pos = new_pos
                self._update_obj_in_model(self.selected)
                self.picked.emit(self.selected)
                self._render(force=True)

    def mouseReleaseEvent(self, event):
        self.drag_move = False
        self.camera_drag = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0
        self.cam.distance = float(np.clip(self.cam.distance * (0.9 ** delta), 0.05, 5.0))

    def keyPressEvent(self, event):
        self.keys_down.add(event.key())
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        self.keys_down.discard(event.key())
        super().keyReleaseEvent(event)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scene Editor 3D (MuJoCo)")
        self.model: Dict[int, SceneObject] = {}
        self.list = QtWidgets.QListWidget()
        self.list.currentItemChanged.connect(self.on_select)
        self.log_box = QtWidgets.QTextEdit()
        self.log_box.setReadOnly(True)
        self.mode = "select"

        self.viewer = MujocoViewport(self.model)
        self.viewer.picked.connect(self.select_obj)
        self.viewer.mode_getter = lambda: self.mode

        # Toolbar with icons
        style = self.style()
        self.select_btn = self._tool_button("Select", style.standardIcon(QtWidgets.QStyle.SP_ArrowBack), True)
        self.move_btn = self._tool_button("Move", style.standardIcon(QtWidgets.QStyle.SP_ArrowUp))
        self.rotate_btn = self._tool_button("Rotate", style.standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        self.camera_btn = self._tool_button("Camera", style.standardIcon(QtWidgets.QStyle.SP_ComputerIcon))
        self.reset_cam_btn = QtWidgets.QToolButton(text="Reset Cam")
        self.reset_cam_btn.clicked.connect(self.reset_camera)

        self.mode_group = QtWidgets.QButtonGroup(self)
        for btn in [self.select_btn, self.move_btn, self.rotate_btn, self.camera_btn]:
            self.mode_group.addButton(btn)
        self.mode_group.setExclusive(True)
        self.mode_group.buttonClicked.connect(self.on_mode_change)

        add_cube = QtWidgets.QToolButton(text="Add Cube")
        add_cube.setIcon(style.standardIcon(QtWidgets.QStyle.SP_FileDialogNewFolder))
        add_cube.clicked.connect(self.add_cube)
        add_goal = QtWidgets.QToolButton(text="Add Goal")
        add_goal.setIcon(style.standardIcon(QtWidgets.QStyle.SP_DialogYesButton))
        add_goal.clicked.connect(self.add_goal)
        del_btn = QtWidgets.QToolButton(text="Delete")
        del_btn.setIcon(style.standardIcon(QtWidgets.QStyle.SP_TrashIcon))
        del_btn.clicked.connect(self.delete_selected)
        color_btn = QtWidgets.QToolButton(text="Color")
        color_btn.setIcon(style.standardIcon(QtWidgets.QStyle.SP_DriveDVDIcon))
        color_btn.clicked.connect(self.set_color)
        play_btn = QtWidgets.QToolButton(text="Play (simulate)")
        play_btn.setIcon(style.standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        play_btn.setCheckable(True)
        play_btn.clicked.connect(self.play)
        self.play_btn = play_btn

        # Nudge controls
        self.move_step = QtWidgets.QDoubleSpinBox()
        self.move_step.setRange(0.001, 0.1)
        self.move_step.setDecimals(3)
        self.move_step.setSingleStep(0.001)
        self.move_step.setValue(0.01)
        self.rot_step = QtWidgets.QDoubleSpinBox()
        self.rot_step.setRange(1, 90)
        self.rot_step.setSingleStep(1)
        self.rot_step.setValue(5)

        nudge_layout = QtWidgets.QGridLayout()
        nudge_layout.addWidget(QtWidgets.QLabel("Nudge:"), 0, 0, 1, 2)
        nudge_layout.addWidget(self._nudge_btn("↑", lambda: self.viewer.nudge(1, +1, self.move_step.value())), 1, 1)
        nudge_layout.addWidget(self._nudge_btn("↓", lambda: self.viewer.nudge(1, -1, self.move_step.value())), 3, 1)
        nudge_layout.addWidget(self._nudge_btn("←", lambda: self.viewer.nudge(0, -1, self.move_step.value())), 2, 0)
        nudge_layout.addWidget(self._nudge_btn("→", lambda: self.viewer.nudge(0, +1, self.move_step.value())), 2, 2)
        nudge_layout.addWidget(self._nudge_btn("Z+", lambda: self.viewer.nudge(2, +1, self.move_step.value())), 1, 3)
        nudge_layout.addWidget(self._nudge_btn("Z-", lambda: self.viewer.nudge(2, -1, self.move_step.value())), 3, 3)
        nudge_layout.addWidget(QtWidgets.QLabel("Move step (m)"), 4, 0, 1, 2)
        nudge_layout.addWidget(self.move_step, 4, 2, 1, 2)
        nudge_layout.addWidget(QtWidgets.QLabel("Rot step (deg)"), 5, 0, 1, 2)
        nudge_layout.addWidget(self.rot_step, 5, 2, 1, 2)
        nudge_layout.addWidget(self._nudge_btn("Yaw +", lambda: self.viewer.nudge_rot(2, +1, self.rot_step.value())), 6, 0, 1, 2)
        nudge_layout.addWidget(self._nudge_btn("Yaw -", lambda: self.viewer.nudge_rot(2, -1, self.rot_step.value())), 6, 2, 1, 2)

        self.pos_spin = [
            self._spin_box(WORK_X[0], WORK_X[1]),
            self._spin_box(WORK_Y[0], WORK_Y[1]),
            self._spin_box(WORK_Z[0], WORK_Z[1]),
        ]
        self.rot_spin = [
            self._spin_box(-180, 180, step=5),
            self._spin_box(-180, 180, step=5),
            self._spin_box(-180, 180, step=5),
        ]
        for s in self.pos_spin + self.rot_spin:
            s.valueChanged.connect(self.apply_transform)

        transform_form = QtWidgets.QFormLayout()
        transform_form.addRow("Position X/Y/Z (m)", self._hbox(self.pos_spin))
        transform_form.addRow("Rotation X/Y/Z (deg)", self._hbox(self.rot_spin))

        tool_row = self._hbox(
            [
                self.select_btn,
                self.move_btn,
                self.rotate_btn,
                self.camera_btn,
                self.reset_cam_btn,
                add_cube,
                add_goal,
                del_btn,
                color_btn,
                play_btn,
            ]
        )

        self.select_btn.setChecked(True)

        side = QtWidgets.QVBoxLayout()
        side.addLayout(tool_row)
        side.addWidget(QtWidgets.QLabel("Scene objects"))
        side.addWidget(self.list, 1)
        side.addWidget(QtWidgets.QLabel("Transform"))
        side.addLayout(transform_form)
        side.addLayout(nudge_layout)
        side.addWidget(QtWidgets.QLabel("Log"))
        side.addWidget(self.log_box, 1)

        main = QtWidgets.QHBoxLayout(self)
        main.addWidget(self.viewer, 3)
        main.addLayout(side, 2)

        self._refresh_list()
        if not any(o.kind == "goal" for o in self.model.values()):
            self.add_goal(initial=True)

        self.setStyleSheet(
            """
            QWidget { background: #0f1116; color: #e8ebf0; }
            QToolButton, QPushButton {
                background: #1c2028; border: 1px solid #2b313c; padding: 6px 10px; border-radius: 6px;
            }
            QToolButton:checked { background: #3b7bff; color: white; border-color: #4b89ff; }
            QListWidget, QTextEdit {
                background: #0c0e12; border: 1px solid #222732; color: #d8dde5;
            }
            QDoubleSpinBox { background: #12151b; border: 1px solid #222732; }
            """
        )

    # --- Helpers --------------------------------------------------------
    def _tool_button(self, text: str, icon: QtGui.QIcon, checked: bool = False):
        btn = QtWidgets.QToolButton()
        btn.setText(text)
        btn.setIcon(icon)
        btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        btn.setCheckable(True)
        btn.setChecked(checked)
        return btn

    def _nudge_btn(self, text: str, fn):
        btn = QtWidgets.QToolButton()
        btn.setText(text)
        btn.clicked.connect(fn)
        return btn

    def _spin_box(self, minimum: float, maximum: float, step: float = 0.01):
        sb = QtWidgets.QDoubleSpinBox()
        sb.setRange(minimum - 0.05, maximum + 0.05)
        sb.setSingleStep(step)
        sb.setDecimals(4)
        return sb

    def _hbox(self, widgets: List[QtWidgets.QWidget]):
        h = QtWidgets.QHBoxLayout()
        for w in widgets:
            h.addWidget(w)
        return h

    # --- UI actions ------------------------------------------------------
    def add_cube(self):
        obj = SceneObject("cube", f"Cube#{_new_id()}", np.array([0.05, 0.5, DEFAULT_CUBE_Z]), np.zeros(3))
        obj.body_name = f"cube_{_new_id()}"
        obj.geom_name = f"{obj.body_name}_geom"
        self.model[id(obj)] = obj
        self.viewer._rebuild_model()
        self._refresh_list(select=obj)

    def add_goal(self, initial: bool = False):
        obj = SceneObject(
            "goal",
            f"Goal#{_new_id()}",
            np.array([0.15, 0.75, DEFAULT_GOAL_Z]),
            np.zeros(3),
            QtGui.QColor(255, 215, 0),
            geom_name="",
            body_name="",
        )
        self.model[id(obj)] = obj
        self.viewer._rebuild_model()
        self._refresh_list(select=obj)
        if initial:
            self.list.setCurrentItem(self.list.item(self.list.count() - 1))

    def delete_selected(self):
        item = self.list.currentItem()
        if not item:
            return
        obj = item.data(QtCore.Qt.UserRole)
        self.model.pop(id(obj), None)
        self.viewer._rebuild_model()
        self._refresh_list()

    def set_color(self):
        item = self.list.currentItem()
        if not item:
            return
        obj = item.data(QtCore.Qt.UserRole)
        color = QtWidgets.QColorDialog.getColor(obj.color, self)
        if color.isValid():
            obj.color = color
            self.viewer._update_obj_in_model(obj)
            self.viewer._render(force=True)

    def on_select(self, curr, prev=None):
        if not curr:
            return
        obj = curr.data(QtCore.Qt.UserRole)
        for i, s in enumerate(self.pos_spin):
            s.blockSignals(True)
            s.setValue(obj.pos[i])
            s.blockSignals(False)
        for i, s in enumerate(self.rot_spin):
            s.blockSignals(True)
            s.setValue(obj.rot[i])
            s.blockSignals(False)
        self.viewer.set_selected(obj)

    def apply_transform(self):
        item = self.list.currentItem()
        if not item:
            return
        obj = item.data(QtCore.Qt.UserRole)
        obj.pos = _clamp_workspace(np.array([s.value() for s in self.pos_spin]))
        obj.rot = np.array([s.value() for s in self.rot_spin])
        self.viewer._update_obj_in_model(obj)
        self.viewer._render(force=True)

    def play(self):
        if not self.play_btn.isChecked():
            self.viewer.stop_sim()
            self.viewer.stop_policy_sim()
            self.play_btn.setText("Play (simulate)")
            self.play_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
            return
        # stop any running sim and launch policy-controlled sim
        self.viewer.stop_sim()
        self.viewer.stop_policy_sim()
        self.viewer.start_policy_sim(list(self.model.values()))
        self.play_btn.setText("Stop (simulate)")
        self.play_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaStop))

    def select_obj(self, obj: SceneObject):
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.data(QtCore.Qt.UserRole) is obj:
                self.list.setCurrentItem(item)
                break
        self.viewer.set_selected(obj)
        self.viewer._render(force=True)

    def on_mode_change(self, btn):
        mode_text = btn.text().lower()
        if "camera" in mode_text:
            mode = "camera"
        else:
            mode = mode_text
        self.mode = mode

    def reset_camera(self):
        self.viewer.cam.azimuth = -135
        self.viewer.cam.elevation = 30
        self.viewer.cam.distance = 1.35
        self.viewer.cam.lookat = np.array([0.15, 0.6, 0.04])
        self.camera_btn.setChecked(False)
        self.select_btn.setChecked(True)
        self.mode = "select"
        if self.list.count():
            self.list.setCurrentRow(0)

    def _refresh_list(self, select: Optional[SceneObject] = None):
        self.list.clear()
        for obj in self.model.values():
            item = QtWidgets.QListWidgetItem(obj.name)
            item.setData(QtCore.Qt.UserRole, obj)
            self.list.addItem(item)
            if select and obj is select:
                self.list.setCurrentItem(item)

    def log(self, msg: str):
        self.log_box.append(msg)
        self.log_box.ensureCursorVisible()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.resize(1500, 900)
    win.viewer._render(force=True)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

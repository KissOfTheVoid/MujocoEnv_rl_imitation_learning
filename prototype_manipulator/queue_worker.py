"""
Очередь действий -> исполнение в симуляции/манипуляторе.

Как работает:
- опрашивает веб-API /actions/next (prototype_web_interface/app.py)
- для каждого action вызывает execute_action, где вы можете подключить свою политику/скрипт.

Для демонстрации execute_action сейчас просто печатает действие.
"""

import os
import time
import requests
import subprocess
import glob
import shutil
from pathlib import Path

# URL веб-интерфейса (можно переопределить переменной окружения SERVER_URL)
SERVER = os.environ.get("SERVER_URL", "http://127.0.0.1:5000")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "1.0"))  # секунд

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "prototype_web_interface" / "static"
# default to trained checkpoint bundled in experiments/checkpoints/mix_cube; can be overridden via CKPT_DIR env
CKPT_DIR = Path(os.environ.get("CKPT_DIR", ROOT / "experiments" / "checkpoints" / "mix_cube"))
EVAL_ROLLOUTS = int(os.environ.get("EVAL_ROLLOUTS", "1"))


def run_policy_eval():
    """Запуск eval, сохранение видео в static/latest.mp4."""
    if not CKPT_DIR.exists():
        print(f"[EXECUTOR] CKPT_DIR не найден: {CKPT_DIR}")
        return
    cmd = [
        "python",
        str(ROOT / "experiments" / "imitate_episodes.py"),
        "--eval",
        "--task_name",
        "mix_cube",
        "--ckpt_dir",
        str(CKPT_DIR),
        "--policy_class",
        "ACT",
        "--batch_size",
        "96",
        "--seed",
        "0",
        "--num_epochs",
        "2177",
        "--lr",
        "1e-5",
        "--kl_weight",
        "10",
        "--chunk_size",
        "100",
        "--hidden_dim",
        "512",
        "--dim_feedforward",
        "3200",
        "--temporal_agg",
        "--num_rollouts",
        str(EVAL_ROLLOUTS),
    ]
    print(f"[EXECUTOR] Запуск eval: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # safer on Apple Silicon
    try:
        subprocess.run(cmd, cwd=ROOT, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"[EXECUTOR] Eval завершился с ошибкой: {exc}")
        return

    videos = sorted(glob.glob(str(CKPT_DIR / "video*.mp4")), key=os.path.getmtime, reverse=True)
    if not videos:
        print("[EXECUTOR] Видео не найдено после eval")
        return
    latest = videos[0]
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    dst = STATIC_DIR / "latest.mp4"
    shutil.copyfile(latest, dst)
    print(f"[EXECUTOR] Видео сохранено: {dst}")


def execute_action(action: dict):
    """
    Пример интеграции: для завершающей команды запускаем eval и кладём видео в /static/latest.mp4
    """
    print(f"[EXECUTOR] Выполняю: {action}")
    action_type = action.get("action_type")
    if action_type in {"place", "throw", "move_to_point"}:
        run_policy_eval()
    time.sleep(0.2)
    print(f"[EXECUTOR] Готово: {action}")


def poll_loop():
    print(f"[WORKER] Старт опроса {SERVER}")
    while True:
        try:
            resp = requests.post(f"{SERVER}/actions/next", timeout=5)
            if resp.status_code == 204:
                time.sleep(POLL_INTERVAL)
                continue
            resp.raise_for_status()
            action = resp.json()
            execute_action(action)
        except Exception as exc:
            print(f"[WORKER] Ошибка: {exc}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    poll_loop()

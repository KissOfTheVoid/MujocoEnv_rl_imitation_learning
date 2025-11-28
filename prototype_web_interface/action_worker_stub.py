"""
Пример воркера, который может забирать задания из веб-очереди (/actions/next)
и передавать их в свой исполнитель (например, ACT-политику или скриптованную логику).

Это скелет: замените execute_action(...) на вызов вашего манипулятора.
"""

import requests
import time

SERVER = "http://localhost:5000"  # URL вашего Flask сервера
POLL_INTERVAL = 1.0  # секунд


def execute_action(action: dict):
    """
    TODO: здесь подключите нейронку/манипулятор.
    action = {"action_type": "...", "target": [x,y,z] | None, ...}
    """
    print(f"[EXECUTOR] Выполняю: {action}")
    # Пример: if action["action_type"] == "pick": run_policy_pick(...)
    time.sleep(0.5)  # имитация выполнения
    print(f"[EXECUTOR] Готово: {action}")


def poll_loop():
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
            print(f"[EXECUTOR] Ошибка: {exc}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    poll_loop()

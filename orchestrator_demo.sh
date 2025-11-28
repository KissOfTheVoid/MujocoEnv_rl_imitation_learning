#!/usr/bin/env bash
# Быстрый запуск веб-UI и воркера очереди в отдельных процессах (локально).
# Запускать из корня репозитория.
set -e

WEB_DIR="prototype_web_interface"
WORKER="prototype_manipulator/queue_worker.py"
HOST="127.0.0.1"
PORT="${PORT:-64562}"

if [ ! -d "$WEB_DIR" ]; then
  echo "Не найден $WEB_DIR; запускайте из корня"
  exit 1
fi

# проверяем занятость порта
is_busy() { lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
if is_busy "$PORT"; then
  echo "Порт $PORT занят. Освободите его или укажите другой в переменной PORT."
  exit 1
fi

echo "Запуск Flask UI (http://${HOST}:${PORT})..."
(
  cd "$WEB_DIR"
  FLASK_APP=app.py flask run --host "$HOST" --port "$PORT"
) &
FLASK_PID=$!

# ждём пока порт станет LISTEN
for _ in {1..30}; do
  if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

echo "Запуск воркера очереди..."
SERVER_URL="http://${HOST}:${PORT}" python "$WORKER" &
WORKER_PID=$!

trap "kill $FLASK_PID $WORKER_PID 2>/dev/null" INT TERM EXIT

wait
echo "Порт UI: $PORT"

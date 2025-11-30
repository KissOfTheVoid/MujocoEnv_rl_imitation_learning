# ACT: Action Chunking with Transformers for Robotic Manipulation

Imitation learning система для бимануальной манипуляции объектами на базе ALOHA робота в симуляции MuJoCo.

## Требования

- NVIDIA GPU с CUDA **или** Apple Silicon (M-серии) с Metal/MPS; на CPU/MPS обучение будет медленнее, но работает
- Ubuntu 20.04+ / Linux или macOS 13+ (ARM64)
- Python 3.10
- ~50GB свободного места для датасетов

## Установка

### 1. Клонирование репозитория

```bash
git clone git@github.com:KissOfTheVoid/MujocoEnv_rl_imitation_learning.git
cd MujocoEnv_rl_imitation_learning
```

### 2. Создание conda окружения

```bash
conda create -n act python=3.10 -y
conda activate act
```

### 3. Установка PyTorch

- **Linux + NVIDIA CUDA**:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- **Apple Silicon (MPS/Metal)**:
  ```bash
  pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
  # для автопадения на CPU, если MPS чего-то не поддерживает
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```

### 4. Установка зависимостей

```bash
pip install dm_control mujoco pyquaternion pyyaml rospkg pexpect mujoco_py opencv-python matplotlib einops packaging h5py ipython tqdm tensorboard
cd experiments/detr && pip install -e . && cd ../..
```

### 5. Настройка путей

Отредактируйте `experiments/constants.py`:
```python
DATA_DIR = '/path/to/your/experiments/raw_data'
```

### 6. Настройка рендеринга

- Linux/сервер без дисплея:
  ```bash
  export MUJOCO_GL=egl
  ```
- macOS + Apple Silicon: используйте стандартный GLFW/Metal (`export MUJOCO_GL=glfw`). Headless EGL недоступен.

Добавьте в `~/.bashrc` (или `~/.zshrc`) для постоянного применения.

## Генерация датасета

### Доступные задачи

| Задача | Описание | Рекомендуемое кол-во эпизодов |
|--------|----------|-------------------------------|
| `single_cube` | Один куб | 50 |
| `single_torus` | Один тор | 50 |
| `multiple_red` | Несколько красных объектов | 50 |
| `multiple_green` | Несколько зелёных объектов | 50 |
| `multiple_blue` | Несколько синих объектов | 50 |
| `mix_cube` | Смешанные кубы (зелёные + красные) | 100-800 |

### Запуск генерации

```bash
cd experiments
python record_sim_episodes.py --task_name mix_cube --dataset_dir ./raw_data/mix_cube --num_episodes 400
```

## Обучение

### Базовый запуск

```bash
cd experiments
python imitate_episodes.py \
    --task_name mix_cube \
    --ckpt_dir ./checkpoints/mix_cube \
    --policy_class ACT \
    --batch_size 512 \
    --seed 0 \
    --num_epochs 10000 \
    --lr 1e-5 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --temporal_agg
```

### Параметры

| Параметр | Описание | Рекомендуемое значение |
|----------|----------|------------------------|
| `--batch_size` | Размер батча (зависит от GPU памяти) | 512 (A100 80GB), 256 (A100 40GB), 128 (RTX 3090) |
| `--num_epochs` | Максимальное число эпох | 10000 (с early stopping) |
| `--lr` | Learning rate | 1e-5 |
| `--kl_weight` | Вес KL-дивергенции | 10 |
| `--chunk_size` | Размер чанка действий | 100 |
| `--temporal_agg` | Темпоральная агрегация | Рекомендуется включить |

### Особенности обучения

- **Mixed Precision Training (AMP)** - автоматически включен для ускорения
- **Early Stopping** - patience=200 эпох без улучшения val_loss
- **TensorBoard** - логи сохраняются в `ckpt_dir/tensorboard/`
- **Чекпоинты** - сохраняются каждые 100 эпох + лучшая модель `policy_best.ckpt`

### Мониторинг обучения

```bash
# TensorBoard
tensorboard --logdir ./checkpoints/mix_cube/tensorboard

# Логи в реальном времени (при запуске через nohup)
tail -f /tmp/train.log
```

## Evaluation

### Запуск оценки с генерацией видео

```bash
python imitate_episodes.py \
    --eval \
    --task_name mix_cube \
    --ckpt_dir ./checkpoints/mix_cube \
    --policy_class ACT \
    --batch_size 512 \
    --seed 0 \
    --num_epochs 2177 \
    --lr 1e-5 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --temporal_agg \
    --resume_ckpt ./checkpoints/mix_cube/policy_best.ckpt  # для продолжения обучения/дообучения
```

Параметр `--num_epochs` указывает эпоху чекпоинта для загрузки.

### Результаты

Видео сохраняются в `ckpt_dir/video*.mp4`. Evaluation проводится на 50 эпизодах (25 на каждый подтаск).

## Результаты экспериментов

| Датасет | Эпизодов | Val Loss | Success Rate (green/red) |
|---------|----------|----------|--------------------------|
| mix_cube | 100 | 0.133 | 52% / 64% |
| mix_cube_400 | 400 | 0.083 | 96% / 94% |

## Структура проекта

```
experiments/
├── assets/              # MuJoCo XML модели роботов и сцен
├── checkpoints/         # Сохранённые модели (исключено из git)
├── raw_data/            # Датасеты (исключено из git)
├── detr/                # Transformer архитектура
├── constants.py         # Конфигурация задач и путей
├── imitate_episodes.py  # Основной скрипт обучения/eval
├── record_sim_episodes.py # Генерация датасета
├── sim_env.py           # Симуляция окружения
└── utils.py             # Вспомогательные функции
```

## Troubleshooting

### CUDA Out of Memory
Уменьшите `--batch_size` (512 → 256 → 128)

### EGL/Rendering ошибки
```bash
export MUJOCO_GL=egl
# или
export MUJOCO_GL=osmesa
```

### Ошибка "env not found" при eval
Убедитесь что task_name совпадает с ключом в `sim_env.py` (например, используйте `mix_cube` вместо `mix_cube_400` для eval)

## Быстрый запуск веб-интерфейса с очередью действий

- Запуск Flask UI и воркера очереди (локально):
  ```bash
  ./orchestrator_demo.sh
  ```
  UI: http://127.0.0.1:5000 — можно ставить объекты на 3D-сцене, отправлять действия в очередь, скачивать CSV сцены/заказа.
- Воркеры:
  - `prototype_web_interface/action_worker_stub.py` — пример опроса очереди.
  - `prototype_manipulator/queue_worker.py` — опрос очереди, сюда подключайте реальный исполнитель/нейронку.

## Ссылки

- [ACT Paper](https://arxiv.org/abs/2304.13705) - Action Chunking with Transformers
- [ALOHA Project](https://tonyzhaozh.github.io/aloha/) - Hardware platform
- [Original Repository](https://github.com/BatarchiZ/bachelors_diploma) - Исходный код диплома

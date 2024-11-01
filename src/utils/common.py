import random
import numpy as np
import torch
import yaml

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_experiment_result(file_path, exp_name, model_name, batch_size, epochs, ATE):
    new_entry = {
        exp_name: {
            "model_name": model_name,
            "batch_size": batch_size,
            "epochs": epochs,
            "ATE": ATE
        }
    }

    try:
        # 既存の結果を読み込む
        with open(file_path, 'r') as file:
            existing_data = yaml.safe_load(file)
            if existing_data is None:
                existing_data = {}
    except FileNotFoundError:
        existing_data = {}

    # 新しい結果を追加
    existing_data.update(new_entry)

    # 変更を保存
    with open(file_path, 'w') as file:
        yaml.dump(existing_data, file)

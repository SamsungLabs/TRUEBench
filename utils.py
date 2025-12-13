import json
import os


def get_configs(config_name: str) -> dict:
    current_dir = os.path.dirname(__file__)
    configs_folder = os.path.join(current_dir, "configs")
    config_path = os.path.join(configs_folder, f"{config_name}.json")

    assert os.path.exists(config_path), f"{config_path} file does not exist."

    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    return configs


def get_model_configs(config: str) -> dict:
    try:
        model_configs = json.loads(config)
    except json.JSONDecodeError as e:
        print("Read model config: ", config)
        model_configs = get_configs(config)
    return model_configs


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} does not exists. Create directory.")
    else:
        print(f"{path} already exists.")

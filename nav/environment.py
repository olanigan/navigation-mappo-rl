from config_models import *
import yaml

if __name__ == "__main__":
    config_file = "config/basic_env.yaml"

    with open(config_file, 'r') as f:
        config_data = yaml.safe_load(f)

    env_config = EnvConfig(**config_data)

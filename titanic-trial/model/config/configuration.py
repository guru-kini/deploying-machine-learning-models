from pathlib import Path
from typing import List

from pydantic import BaseModel
from strictyaml import YAML, load

import model

PACKAGE_ROOT = Path(model.__file__).resolve().parent
DATASET_DIR = PACKAGE_ROOT / "data"
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

class AppConfig(BaseModel) :
    """
    Application Specific Configuration
    """
    package_name: str
    seed: int
    pipeline_file: str
    
class ModelConfig(BaseModel) :
    """
    Model Specific Configuration
    """
    numerical_variables: List[str]
    categorical_variables: List[str]
    cabin_var: List[str]
    target: str
    test_size: float
    
class Config(BaseModel) :
    app_config: AppConfig
    model_config: ModelConfig
    
def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config() -> YAML: 
    config_path = find_config_file()
    
    if config_path: 
        with open(config_path, "r") as config_file: 
            return load(config_file.read())
    raise OSError("No config file found in path " + config_path)

def create_config() -> Config: 
    yaml = fetch_config()
    
    return Config(
        app_config=AppConfig(**yaml.data),
        model_config=ModelConfig(**yaml.data)
    )

""" The main export """
_config = create_config()
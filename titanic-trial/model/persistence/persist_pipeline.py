import typing as t

import joblib
import os.path
from sklearn.pipeline import Pipeline
from model.config.configuration import _config, TRAINED_MODEL_DIR
from model import __version__ as MODEL_VERSION, config

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = persisted_model_path()
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    
def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    if(os.path.isfile(file_path)) :
        trained_model = joblib.load(filename=file_path)
        return trained_model
    return None

def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    
    if not os.path.isdir(TRAINED_MODEL_DIR) :
        # Create the folder if doesn't exist
        os.mkdir(TRAINED_MODEL_DIR)
        return
    
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()

def persisted_model_path() -> str: 
    return f"{config.app_config.pipeline_file}{MODEL_VERSION}.pkl"
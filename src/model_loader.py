import tensorflow as tf
import os
from utils.file_utils import ensure_directory
from utils.config import MODEL_TYPE, SERVER_PREFIX, CACHE_DIR

def download_model():
    """
    Download and load the Metrabs model from the specified URL.
    
    Returns:
        str: Path to the downloaded model.
    """
    print(f"[Model Loader] Loading Metrabs model: {MODEL_TYPE}")
    
    model_path = os.path.join(CACHE_DIR, MODEL_TYPE)
    
    # Check if model directory exists
    if os.path.exists(model_path):
        print(f"[Model Loader] Model already exists at: {model_path}")
        return model_path
    
    # Ensure cache directory exists
    ensure_directory(CACHE_DIR)
    print(f"[Model Loader] Cache directory ensured: {CACHE_DIR}")
    
    # Download and extract model
    print(f"[Model Loader] Downloading model from: {SERVER_PREFIX}/{MODEL_TYPE}_20211019.zip")
    model_zippath = tf.keras.utils.get_file(
        origin=f'{SERVER_PREFIX}/{MODEL_TYPE}_20211019.zip',
        extract=True, cache_subdir='models')
    model_path = os.path.join(os.path.dirname(model_zippath), MODEL_TYPE)
    print(f"[Model Loader] Model downloaded and extracted to: {model_path}")
    return model_path
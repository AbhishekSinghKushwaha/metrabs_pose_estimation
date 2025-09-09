import os

def ensure_directory(directory):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory (str): Path to the directory.
    """
    if directory and not os.path.exists(directory):
        print(f"[File Utils] Creating directory: {directory}")
        os.makedirs(directory)
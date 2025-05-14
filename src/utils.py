import os
import glob

def get_dataset_folders(directory):
    """Returns a list of dataset folders inside the 'data/' directory."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"🚨 Directory not found: {directory}")
    
    return [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory, folder))]

def get_xes_file(directory):
    """Finds and returns the first XES file in a dataset folder."""
    xes_files = glob.glob(os.path.join(directory, "*.xes"))
    return xes_files[0] if xes_files else None

def get_pnml_files(directory):
    """Returns a list of PNML files in a dataset folder."""
    return glob.glob(os.path.join(directory, "*.pnml"))
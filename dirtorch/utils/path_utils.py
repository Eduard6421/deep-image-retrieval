import os
from pathlib import Path

def get_root():
    return Path(__file__).parents[5]


def get_data_root():
    return os.path.join(get_root(), 'Datasets')

def get_model_root():
    return os.path.join(get_root(), 'Pretrained_Models')

def get_embedding_root():
    return os.path.join(get_root(), 'Embeddings')

def get_results_root():
    return os.path.join(get_root(), 'Results')

def get_embedding_subfolder():
    return os.path.join(get_embedding_root(), 'DEEP_Image_Retrieval')
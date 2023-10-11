import os

_ROOT = os.path.abspath(os.path.dirname(__file__))

def get_distractor_folder_path():
    return _ROOT


def get_distractor_data_path():
    return os.path.join(_ROOT, 'data')

def get_distractor_data_edge_path():
    return os.path.join(_ROOT, 'data', 'edge')

# def add_data_save_path(path):
#     return os.path.join(_ROOT, path)
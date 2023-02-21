import os

from dirtorch.utils.path_utils import get_data_root
from .generic import ImageListLabels




class Landmarks_clean(ImageListLabels):
    def __init__(self):
        ImageListLabels.__init__(self, os.path.join(get_data_root(), 'landmarks/annotations/annotation_clean_train.txt'),
                                 os.path.join(get_data_root(), 'landmarks/'))

class Landmarks_clean_val(ImageListLabels):
    def __init__(self):
        ImageListLabels.__init__(self, os.path.join(get_data_root(), 'landmarks/annotations/annotation_clean_val.txt'),
                                 os.path.join(get_data_root(), 'landmarks/'))

class Landmarks_lite(ImageListLabels):
    def __init__(self):
        ImageListLabels.__init__(self, os.path.join(get_data_root(), 'landmarks/annotations/extra_landmark_images.txt'),
                                 os.path.join(get_data_root(), 'landmarks/'))

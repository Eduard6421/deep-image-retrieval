from dirtorch.utils.path_utils import get_data_root
from .generic import ImageListRelevants
import os


class Paris6K(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(get_data_root(), 'paris6k/gnd_paris6k.pkl'),
                                 root=os.path.join(get_data_root(), 'paris6k'))

class RParis6K(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(get_data_root(), 'rparis6k/gnd_rparis6k.pkl'),
                                 root=os.path.join(get_data_root(), 'rparis6k'))

        
class RParis6K_Drift(ImageListRelevants):
    def __init__(self):
                ImageListRelevants.__init__(self, os.path.join(get_data_root(), 'rparis6k/gnd_rparis6k_drift_deep.pkl'),
                                 root=os.path.join(get_data_root(), 'rparis6k'))
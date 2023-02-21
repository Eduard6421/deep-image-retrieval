import os

from dirtorch.utils.path_utils import get_data_root
from .generic import ImageListRelevants



class Oxford5K(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(get_data_root(), 'oxford5k/gnd_oxford5k.pkl'),
                                 root=os.path.join(get_data_root(), 'oxford5k'))

class ROxford5K(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(get_data_root(), 'roxford5k/gnd_roxford5k.pkl'),
                                 root=os.path.join(get_data_root(), 'roxford5k'))

        
class ROxford5K_Drift(ImageListRelevants):
    def __init__(self):
                ImageListRelevants.__init__(self, os.path.join(get_data_root(), 'roxford5k/gnd_roxford5k_drift_deep.pkl'),
                                 root=os.path.join(get_data_root(), 'roxford5k'))
from .generic import ImageListRelevants
import os

DB_ROOT = os.environ['DB_ROOT']

class Caltech101_70(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'caltech101/gnd_caltech101_70.pkl'),
                                 root=os.path.join(DB_ROOT, 'caltech101'))
        
class Caltech101_140(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'caltech101/gnd_caltech101_140.pkl'),
                                 root=os.path.join(DB_ROOT, 'caltech101'))
        
class Caltech101_350(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'caltech101/gnd_caltech101_350.pkl'),
                                 root=os.path.join(DB_ROOT, 'caltech101'))
        

class Caltech101_700(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'caltech101/gnd_caltech101_700.pkl'),
                                 root=os.path.join(DB_ROOT, 'caltech101'))
        
class Caltech101_1400(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'caltech101/gnd_caltech101_1400.pkl'),
                                 root=os.path.join(DB_ROOT, 'caltech101'))
        
        
class Caltech101_700_Train(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'caltech101/gnd_caltech101_700_train.pkl'),
                                 root=os.path.join(DB_ROOT, 'caltech101'))
        
class Caltech101_700_Drift(ImageListRelevants):
    def __init__(self):
                ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'caltech101/gnd_caltech101_700_drift_deep.pkl'),
                                 root=os.path.join(DB_ROOT, 'caltech101'))
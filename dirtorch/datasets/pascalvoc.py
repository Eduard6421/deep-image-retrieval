from .generic import ImageListRelevants
import os

DB_ROOT = os.environ['DB_ROOT']

class PascalVOC_70(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'pascalvoc/gnd_pascalvoc_70.pkl'),
                                 root=os.path.join(DB_ROOT, 'pascalvoc'))
        
class PascalVOC_140(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'pascalvoc/gnd_pascalvoc_140.pkl'),
                                 root=os.path.join(DB_ROOT, 'pascalvoc'))
        
class PascalVOC_350(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'pascalvoc/gnd_pascalvoc_350.pkl'),
                                 root=os.path.join(DB_ROOT, 'pascalvoc'))

class PascalVOC_700(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'pascalvoc/gnd_pascalvoc_700.pkl'),
                                 root=os.path.join(DB_ROOT, 'pascalvoc'))
        
class PascalVOC_1400(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'pascalvoc/gnd_pascalvoc_1400.pkl'),
                                 root=os.path.join(DB_ROOT, 'pascalvoc'))
        
class PascalVOC_700_No_Bbx(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'pascalvoc/gnd_pascalvoc_700_no_bbx.pkl'),
                                 root=os.path.join(DB_ROOT, 'pascalvoc'))
        
class PascalVOC_700_Medium(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'pascalvoc/gnd_pascalvoc_700_medium.pkl'),
                                 root=os.path.join(DB_ROOT, 'pascalvoc'))
        
class PascalVOC_700_Train(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'pascalvoc/gnd_pascalvoc_700_train.pkl'),
                                 root=os.path.join(DB_ROOT, 'pascalvoc'))
        
class PascalVOC_700_Medium_Train(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'pascalvoc/gnd_pascalvoc_700_medium_train.pkl'),
                                 root=os.path.join(DB_ROOT, 'pascalvoc')) 
        
class PascalVOC_700_No_Bbx_Train(ImageListRelevants):
    def __init__(self):
        ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'pascalvoc/gnd_pascalvoc_700_no_bbx_train.pkl'),
                                 root=os.path.join(DB_ROOT, 'pascalvoc'))
        
class PascalVOC_700_Drift(ImageListRelevants):
    def __init__(self):
                ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'pascalvoc/gnd_pascalvoc_700_drift_deep.pkl'),
                                 root=os.path.join(DB_ROOT, 'pascalvoc'))
class PascalVOC_700_Medium_Drift(ImageListRelevants):
    def __init__(self):
                ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'pascalvoc/gnd_pascalvoc_700_medium_drift_deep.pkl'),
                                 root=os.path.join(DB_ROOT, 'pascalvoc'))
            
class PascalVOC_700_No_Bbx_Drift(ImageListRelevants):
    def __init__(self):
                ImageListRelevants.__init__(self, os.path.join(DB_ROOT, 'pascalvoc/gnd_pascalvoc_700_no_bbx_drift_deep.pkl'),
                                 root=os.path.join(DB_ROOT, 'pascalvoc'))
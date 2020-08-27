__author__ = 'mhgou, cxwang and hsfang'
__version__ = '1.0'

from .graspnet import GraspNet

class GraspNetEval(GraspNet):
    def __init__(self, root, camera, split):
        super(GraspNetEval, self).__init__(root, camera, split)
        
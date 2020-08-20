__author__ = 'mhgou'
__version__ = '1.0'

# GraspNetAPI example for loading grasp for a scene.
# change the graspnet_root path

####################################################################
graspnet_root = '/DATA1/Benchmark/graspnet' # ROOT PATH FOR GRASPNET
####################################################################

sceneId = 1
from graspnetAPI import GraspNet

# initialize a GraspNet instance  
g = GraspNet(graspnet_root, camera='kinect', split='train')

# load grasp of scene 1 with annotation id = 0, camera = kinect and grasp_thresh = 0.4
_6d_grasp = g.loadGrasp(sceneId = sceneId, annId = 0, format = '6d', camera = 'kinect', grasp_thresh = 0.4)
print('Object ids in scene %d:' % sceneId, _6d_grasp.keys())

for k in _6d_grasp.keys():
    print('=======================\nobject id=%d, grasps number = %d format = 6d\n=======================' % (k,_6d_grasp[k]['depths'].shape[0]))
    print('points:')
    print(_6d_grasp[k]['points'])
    print('Rs:')
    print(_6d_grasp[k]['Rs'])
    print('depths:')
    print(_6d_grasp[k]['depths'])
    print('widths:')
    print(_6d_grasp[k]['widths'])
    print('friction coefficients:')
    print(_6d_grasp[k]['fric_coefs'])

# rect_grasp = g.loadGrasp(sceneId = sceneId, annId = 0, format = '6d', camera = 'kinect', grasp_thresh = 0.4)
# print('=======================\ngrasps number = %d format = rect\n=======================' % (rect_grasp.shape[0],))
# print(rect_grasp)

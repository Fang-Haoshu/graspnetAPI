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
grasp = g.loadGrasp(sceneId = sceneId, annId = 0, camera = 'kinect', grasp_thresh = 0.4)
print('Object ids in scene %d:' % sceneId, grasp.keys())

for k in grasp.keys():
    print('=======================\nobject id=%d, grasps number = %d\n=======================' % (k,grasp[k]['depths'].shape[0]))
    print('points:')
    print(grasp[k]['points'])
    print('Rs:')
    print(grasp[k]['Rs'])
    print('depths:')
    print(grasp[k]['depths'])
    print('widths:')
    print(grasp[k]['widths'])
    print('friction coefficients:')
    print(grasp[k]['fric_coefs'])

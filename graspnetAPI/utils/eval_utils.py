import os
import time
import numpy as np
import open3d as o3d
from transforms3d.euler import euler2mat, quat2mat

from .rotation import batch_viewpoint_params_to_matrix, matrix_to_dexnet_params

from grasp_nms import nms_grasp
from dexnet.grasping.quality import PointGraspMetrics3D
from dexnet.grasping import ParallelJawPtGrasp3D, GraspableObject3D, GraspQualityConfigFactory, Contact3D
from meshpy.obj_file import ObjFile
from meshpy.sdf_file import SdfFile

def get_scene_name(num):
    return ('scene_%04d' % (num,))

def create_table_points(lx, ly, lz, dx=0, dy=0, dz=0, grid_size=0.01):
    xmap = np.linspace(0, lx, int(lx/grid_size))
    ymap = np.linspace(0, ly, int(ly/grid_size))
    zmap = np.linspace(0, lz, int(lz/grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    points = np.stack([xmap, ymap, zmap], axis=-1)
    points = points.reshape([-1, 3])
    return points

def parse_posevector(posevector):
    mat = np.zeros([4,4],dtype=np.float32)
    alpha, beta, gamma = posevector[4:7]
    alpha = alpha / 180.0 * np.pi
    beta = beta / 180.0 * np.pi
    gamma = gamma / 180.0 * np.pi
    mat[:3,:3] = euler2mat(alpha, beta, gamma)
    mat[:3,3] = posevector[1:4]
    mat[3,3] = 1
    obj_idx = int(posevector[0])
    return obj_idx, mat

def load_dexnet_model(data_path):
    '''
    Input:
        data_path: path to load .obj & .sdf files
    Output:
        obj: dexnet model
    '''
    of = ObjFile('{}.obj'.format(data_path))
    sf = SdfFile('{}.sdf'.format(data_path))
    mesh = of.read()
    sdf = sf.read()
    obj = GraspableObject3D(sdf, mesh)
    return obj

def transform_points(points, trans):
    '''
    Input:
        points: (N, 3)
        trans: (4, 4)
    Output:
        points_trans: (N, 3)
    '''
    ones = np.ones([points.shape[0],1], dtype=points.dtype)
    points_ = np.concatenate([points, ones], axis=-1)
    points_ = np.matmul(trans, points_.T).T
    points_trans = points_[:,:3]
    return points_trans

def compute_point_distance(A, B):
    '''
    Input:
        A: (N, 3)
        B: (M, 3)
    Output:
        dists: (N, M)
    '''
    A = A[:, np.newaxis, :]
    B = B[np.newaxis, :, :]
    dists = np.linalg.norm(A-B, axis=-1)
    return dists

def compute_closest_points(A, B):
    '''
    Input:
        A: (N, 3)
        B: (M, 3)
    Output:
        indices: (N,) closest point index in B for each point in A
    '''
    dists = compute_point_distance(A, B)
    indices = np.argmin(dists, axis=-1)
    return indices

def voxel_sample_points(points, voxel_size=0.008):
    '''
    Input:
        points: (N, 3)
    Output:
        points: (n, 3)
    '''
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.array(cloud.points)
    return points

def topk_grasps(grasps, k=10):
    '''
    Input:
        grasps: (N, 11)
        k: int
    Output:
        topk_grasps: (k, 11)
    '''
    assert(k > 0)
    if len(grasps) <= k:
        grasp_confidence = grasps[:,1]
        indices = np.argsort(-grasp_confidence)
        return grasps[indices]
    grasp_confidence = grasps[:, 1]
    topk_indices = np.argsort(-grasp_confidence)[:k]
    topk_grasps = grasps[topk_indices]
    return topk_grasps

def get_grasp_score(grasp, obj, fc_list, force_closure_quality_config):
    tmp, is_force_closure = False, False
    quality = -1
    for ind_, value_fc in enumerate(fc_list):
        value_fc = round(value_fc, 2)
        # print(value_fc)
        tmp = is_force_closure
        is_force_closure = PointGraspMetrics3D.grasp_quality(grasp, obj, force_closure_quality_config[value_fc])
        if tmp and not is_force_closure:
            # print(value_fc)
            quality = round(fc_list[ind_ - 1], 2)
            break
        elif is_force_closure and value_fc == fc_list[-1]:
            # print(value_fc)
            quality = value_fc
            break
        elif value_fc == fc_list[0] and not is_force_closure:
            break
    return quality

def collision_detection(grasp_list, model_list, dexnet_models, poses, scene_points, outlier=0.05, return_contacts=False):
    '''
    Input:
        grasp_list: [(k1,11), (k2,11), ..., (kn,11)] in camera coordinate
        model_list: [(N1, 3), (N2, 3), ..., (Nn, 3)] in camera coordinate
        dexnet_models: [GraspableObject3D,] in model coordinate
        poses: [(4, 4),] from model coordinate to camera coordinate
        scene_points: (Ns, 3) in camera coordinate
    Output:
        collsion_mask_list: [(k1,), (k2,), ..., (kn,)]
        contact_list: [[[ParallelJawPtGrasp3D, Contact3D, Contact3D],],]
            in model coordinate
    '''
    height = 0.02
    depth_base = 0.02
    finger_width = 0.01
    collision_mask_list = list()
    num_models = len(model_list)
    contact_list = list()

    for i in range(num_models):
        if len(grasp_list[i][0]) == 0:
            collision_mask_list.append(list())
            if return_contacts:
                contact_list.append(list())
            continue

        model = model_list[i]
        obj_pose = poses[i]
        dexnet_model = dexnet_models[i]
        grasps = grasp_list[i]
        # print('grasps shape: ', grasps.shape)
        grasp_points = grasps[:, 2:5]
        grasp_towards = grasps[:, 5:8]
        grasp_angles = grasps[:, 8]
        grasp_depths = grasps[:, 9]
        #grasp_widths = grasps[:, 10]
        grasp_widths = np.ones(grasps[:,10].shape)*0.1
        # grasp_points = transform_points(grasp_points, obj_pose)
        # crop scene, remove outlier
        xmin, xmax = model[:,0].min(), model[:,0].max()
        ymin, ymax = model[:,1].min(), model[:,1].max()
        zmin, zmax = model[:,2].min(), model[:,2].max()
        xlim = ((scene_points[:,0] > xmin-outlier) & (scene_points[:,0] < xmax+outlier))
        ylim = ((scene_points[:,1] > ymin-outlier) & (scene_points[:,1] < ymax+outlier))
        zlim = ((scene_points[:,2] > zmin-outlier) & (scene_points[:,2] < zmax+outlier))
        workspace = scene_points[xlim & ylim & zlim]
        # print('workspace shape: ', workspace.shape)
        # print(xmin,xmax,ymin,ymax,zmin,zmax)
        # print(grasp_points)
        # transform scene to gripper frame
        target = (workspace[np.newaxis,:,:] - grasp_points[:,np.newaxis,:])
        #grasp_angles = grasp_angles.reshape(-1)
        grasp_poses = batch_viewpoint_params_to_matrix(grasp_towards, grasp_angles) # gripper to camera coordinate
        # print('target shape 0: ', target.shape)
        target = np.matmul(target, grasp_poses)
        # print('target shape: ', target.shape)
        # collision detection
        mask1 = ((target[:,:,2]>-height/2) & (target[:,:,2]<height/2))
        mask2 = ((target[:,:,0]>-depth_base) & (target[:,:,0]<grasp_depths[:,np.newaxis]))
        mask3 = (target[:,:,1]>-(grasp_widths[:,np.newaxis]/2+finger_width))
        mask4 = (target[:,:,1]<-grasp_widths[:,np.newaxis]/2)
        mask5 = (target[:,:,1]<(grasp_widths[:,np.newaxis]/2+finger_width))
        mask6 = (target[:,:,1]>grasp_widths[:,np.newaxis]/2)
        mask7 = ((target[:,:,0]>-(depth_base+finger_width)) & (target[:,:,0]<-depth_base))
        # print('single mask 1-7 sum:', np.sum(mask1), np.sum(mask2), np.sum(mask3), np.sum(mask4), np.sum(mask5), np.sum(mask6), np.sum(mask7))
        left_mask = (mask1 & mask2 & mask3 & mask4)
        right_mask = (mask1 & mask2 & mask5 & mask6)
        bottom_mask = (mask1 & mask3 & mask5 & mask7)
        collision_mask = np.any((left_mask | right_mask | bottom_mask), axis=-1)
        collision_mask_list.append(collision_mask)

        if return_contacts:
            contacts = list()
            points_in_gripper_mask = (mask1 & mask2 &(~mask4) & (~mask6))
            # print('mask sum ', np.sum(points_in_gripper_mask))
            for grasp_id,_ in enumerate(grasps):
                grasp_point = grasp_points[grasp_id]
                R = grasp_poses[grasp_id]
                width = grasp_widths[grasp_id]
                depth = grasp_depths[grasp_id]
                points_in_gripper = target[grasp_id][points_in_gripper_mask[grasp_id]]
                # print('points in gripper: ', points_in_gripper.shape)
                if len(points_in_gripper) < 10:
                    contacts.append(None)
                    continue
                c1_ind = np.argmin(points_in_gripper[:, 1])
                c2_ind = np.argmax(points_in_gripper[:, 1])
                c1 = points_in_gripper[c1_ind].reshape([3, 1]) # gripper coordinate
                c2 = points_in_gripper[c2_ind].reshape([3, 1])
                # print('contacts before trans', c1, c2)
                c1 = np.dot(R, c1).reshape([3]) + grasp_point # camera coordinate
                c1 = transform_points(c1[np.newaxis,:], np.linalg.inv(obj_pose)).reshape([3]) # model coordinate
                c2 = np.dot(R, c2).reshape([3])+ grasp_point # camera coordinate
                c2 = transform_points(c2[np.newaxis,:], np.linalg.inv(obj_pose)).reshape([3]) # model coordinate
                # print('contacts after trans', c1, c2)
                center = np.array([depth, 0, 0]).reshape([3, 1]) # gripper coordinate
                center = np.dot(grasp_poses[grasp_id], center).reshape([3])
                center = (center + grasp_point).reshape([1,3]) # camera coordinate
                center = transform_points(center, np.linalg.inv(obj_pose)).reshape([3]) # model coordinate
                R = np.dot(obj_pose[:3,:3].T, R)
                binormal, approach_angle = matrix_to_dexnet_params(R)
                grasp = ParallelJawPtGrasp3D(ParallelJawPtGrasp3D.configuration_from_params(
                                            center, binormal, width, approach_angle), depth)
                contact1 = Contact3D(dexnet_model, c1, in_direction=binormal)
                contact2 = Contact3D(dexnet_model, c2, in_direction=-binormal)
                # print((c2-c1)/np.linalg.norm((c2-c1)), binormal)
                contacts.append((grasp, contact1, contact2))
        contact_list.append(contacts)
    
    if return_contacts:
        return collision_mask_list, contact_list
    else:
        return collision_mask_list

def eval_grasp(grasps, models, dexnet_models, poses, config, table=None, voxel_size=0.008):
    '''
        models: in model coordinate
        poses: from model to camera coordinate
        table: in camera coordinate
    '''
    num_models = len(models)
    ## grasp nms
    tic = time.time()
    grasps = np.array(nms_grasp(grasps, 0.1, 30.0/180*np.pi))
    # print(grasps.shape)
    toc = time.time()
    # print('nms time: %f' % (toc-tic))

    ## assign grasps to object
    # merge and sample scene
    tic = time.time()
    model_trans_list = list()
    # model_list = list()
    seg_mask = list()
    for i,model in enumerate(models):
        model_trans = transform_points(model, poses[i])
        seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
        model_trans_list.append(model_trans)
        seg_mask.append(seg)
    seg_mask = np.concatenate(seg_mask, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)
    toc = time.time()
    # print('pre-assign time: %f' % (toc-tic))
    # scene = np.concatenate(model_list, axis=0)
    # assign grasps
    tic = time.time()
    indices = compute_closest_points(grasps[:,2:5], scene)
    model_to_grasp = seg_mask[indices]
    grasp_list = list()
    for i in range(num_models):
        grasp_i = grasps[model_to_grasp==i]
        if len(grasp_i) == 0:
            grasp_list.append(np.array([[]]))
            continue
        grasp_i = topk_grasps(grasp_i, k=10)
        grasp_list.append(grasp_i)
        # print(grasp_list)
    toc = time.time()
    # print('grasp assigning time: %f' % (toc-tic))

    ## collision detection
    tic = time.time()
    if table is not None:
        scene = np.concatenate([scene, table])
    toc = time.time()
    # print('pre detection time: %f' % (toc-tic))
    tic = time.time()
    collision_mask_list, contact_list = collision_detection(
        grasp_list, model_trans_list, dexnet_models, poses, scene, outlier=0.05, return_contacts=True)
    toc = time.time()
    # print('collision detection time: %f' % (toc-tic))
    
    ## evaluate grasps
    # score configurations
    force_closure_quality_config = dict()
    fc_list = np.array([1.2,1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    for value_fc in fc_list:
        value_fc = round(value_fc, 2)
        config['metrics']['force_closure']['friction_coef'] = value_fc
        force_closure_quality_config[value_fc] = GraspQualityConfigFactory.create_config(config['metrics']['force_closure'])
    # get grasp scores
    score_list = list()
    tic = time.time()
    for i in range(num_models):
        dexnet_model = dexnet_models[i]
        collision_mask = collision_mask_list[i]
        contacts = contact_list[i]
        scores = list()
        num_grasps = len(contacts)
        for grasp_id in range(num_grasps):
            if collision_mask[grasp_id]:
                scores.append(-1.)
                continue
            if contacts[grasp_id] is None:
                scores.append(-1.)
                continue
            grasp, c1, c2 = contacts[grasp_id]
            score = get_grasp_score(grasp, dexnet_model, fc_list, force_closure_quality_config)
            scores.append(score)
            #print(score)
        score_list.append(np.array(scores))
    toc = time.time()
    # print('grasp evaluation time: %f' % (toc-tic))
    # print(score_list, collision_mask_list)
    return grasp_list, score_list, collision_mask_list

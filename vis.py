import os
import time
import numpy as np
import open3d as o3d
from transforms3d.euler import euler2mat, quat2mat
from utils import generate_scene_model, generate_scene_pointcloud, generate_views, get_model_grasps, plot_gripper_pro_max, transform_points
from rotation import viewpoint_params_to_matrix, batch_viewpoint_params_to_matrix

def create_table_cloud(width, height, depth, dx=0, dy=0, dz=0, grid_size=0.01):
    xmap = np.linspace(0, width, int(width/grid_size))
    ymap = np.linspace(0, depth, int(depth/grid_size))
    zmap = np.linspace(0, height, int(height/grid_size))
    xmap, ymap, zmap = np.meshgrid(xmap, ymap, zmap, indexing='xy')
    xmap += dx
    ymap += dy
    zmap += dz
    # points = np.stack([xmap, ymap, zmap], axis=-1)
    points = np.stack([xmap, -ymap, -zmap], axis=-1)
    points = points.reshape([-1, 3])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    return cloud


def visAnno(dataset_root, scene_name, anno_idx, camera, num_grasp=10, th=0.3, align_to_table=True, max_width=0.08, save_folder='save_fig', show=False):
    model_list, obj_list, pose_list = generate_scene_model(dataset_root, scene_name, anno_idx, return_poses=True, align=align_to_table, camera=camera)
    point_cloud = generate_scene_pointcloud(dataset_root, scene_name, anno_idx, align=align_to_table, camera=camera)

    table = create_table_cloud(1.0, 0.02, 1.0, dx=-0.5, dy=-0.5, dz=0, grid_size=0.01)
    num_views, num_angles, num_depths = 300, 12, 4
    views = generate_views(num_views)
    collision_label = np.load('{}/collision_label/{}/collision_labels.npz'.format(dataset_root,scene_name))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width = 1280, height = 720)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters('param_{}.json'.format(camera))

    if align_to_table:
        cam_pos = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'cam0_wrt_table.npy'))
        param.extrinsic = np.linalg.inv(cam_pos).tolist()

    grippers = []
    for i, (obj_idx, trans) in enumerate(zip(obj_list, pose_list)):
        sampled_points, offsets, scores, _ = get_model_grasps('%s/grasp_label/%03d_labels.npz'%(dataset_root, obj_idx))
        collision = collision_label['arr_{}'.format(i)]

        cnt = 0
        point_inds = np.arange(sampled_points.shape[0])
        np.random.shuffle(point_inds)

        for point_ind in point_inds:
            target_point = sampled_points[point_ind]
            offset = offsets[point_ind]
            score = scores[point_ind]
            view_inds = np.arange(300)
            np.random.shuffle(view_inds)
            flag = False
            for v in view_inds:
                if flag: break
                view = views[v]
                angle_inds = np.arange(12)
                np.random.shuffle(angle_inds)
                for a in angle_inds:
                    if flag: break
                    depth_inds = np.arange(4)
                    np.random.shuffle(depth_inds)
                    for d in depth_inds:
                        if flag: break
                        angle, depth, width = offset[v, a, d]
                        if score[v, a, d] > th or score[v, a, d] < 0:
                            continue
                        if width > max_width:
                            continue
                        if collision[point_ind, v, a, d]:
                            continue
                        R = viewpoint_params_to_matrix(-view, angle)
                        t = transform_points(target_point[np.newaxis,:], trans).squeeze()
                        R = np.dot(trans[:3,:3], R)
                        gripper = plot_gripper_pro_max(t, R, width, depth, 1.1-score[v, a, d])
                        grippers.append(gripper)
                        flag = True
            if flag:
                cnt += 1
            if cnt == num_grasp:
                break

    vis.add_geometry(point_cloud)
    for gripper in grippers:
        vis.add_geometry(gripper)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    filename = os.path.join(save_folder, '{}_{}_pointcloud.png'.format(scene_name, camera))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    vis.capture_screen_image(filename, do_render=True)
    if show:
        o3d.visualization.draw_geometries([point_cloud, *grippers])


    vis.remove_geometry(point_cloud)
    vis.add_geometry(table)
    for model in model_list:
        vis.add_geometry(model)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    filename = os.path.join(save_folder, '{}_{}_model.png'.format(scene_name, camera))
    vis.capture_screen_image(filename, do_render=True)
    if show:
        o3d.visualization.draw_geometries([table, *model_list, *grippers])


def vis6D(dataset_root, scene_name, anno_idx, camera, align_to_table=True, save_folder='save_fig', show=False):
    model_list, obj_list, pose_list = generate_scene_model(dataset_root, scene_name, anno_idx, return_poses=True, align=align_to_table, camera=camera)
    point_cloud = generate_scene_pointcloud(dataset_root, scene_name, anno_idx, align=align_to_table, camera=camera)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width = 1280, height = 720)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters('param_{}.json'.format(camera))

    if align_to_table:
        cam_pos = np.load(os.path.join(dataset_root, 'scenes', scene_name, camera, 'cam0_wrt_table.npy'))
        param.extrinsic = np.linalg.inv(cam_pos).tolist()

    vis.add_geometry(point_cloud)
    for model in model_list:
        vis.add_geometry(model)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    filename = os.path.join(save_folder, '{}_{}_6d.png'.format(scene_name, camera))
    vis.capture_screen_image(filename, do_render=True)
    if show:
        o3d.visualization.draw_geometries([point_cloud, *model_list])



def visObjGrasp(dataset_root, obj_idx, num_grasp=10, th=0.5, save_folder='save_fig', show=False):
    plyfile = os.path.join(dataset_root, 'models', '%03d'%obj_idx, 'nontextured.ply')
    model = o3d.io.read_point_cloud(plyfile)

    num_views, num_angles, num_depths = 300, 12, 4
    views = generate_views(num_views)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width = 1280, height = 720)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters('param_{}.json'.format(camera))

    sampled_points, offsets, scores, _ = get_model_grasps('%s/grasp_label/%03d_labels.npz'%(dataset_root, obj_idx))

    cnt = 0
    point_inds = np.arange(sampled_points.shape[0])
    np.random.shuffle(point_inds)
    grippers = []

    for point_ind in point_inds:
        target_point = sampled_points[point_ind]
        offset = offsets[point_ind]
        score = scores[point_ind]
        view_inds = np.arange(300)
        np.random.shuffle(view_inds)
        flag = False
        for v in view_inds:
            if flag: break
            view = views[v]
            angle_inds = np.arange(12)
            np.random.shuffle(angle_inds)
            for a in angle_inds:
                if flag: break
                depth_inds = np.arange(4)
                np.random.shuffle(depth_inds)
                for d in depth_inds:
                    if flag: break
                    angle, depth, width = offset[v, a, d]
                    if score[v, a, d] > th or score[v, a, d] < 0:
                        continue
                    if width > max_width:
                        continue
                    R = viewpoint_params_to_matrix(-view, angle)
                    t = transform_points(target_point[np.newaxis,:], trans).squeeze()
                    R = np.dot(trans[:3,:3], R)
                    gripper = plot_gripper_pro_max(t, R, width, depth, 1.1-score[v, a, d])
                    grippers.append(gripper)
                    flag = True
        if flag:
            cnt += 1
        if cnt == num_grasp:
            break

    vis.add_geometry(model)
    for gripper in grippers:
        vis.add_geometry(gripper)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.poll_events()
    filename = os.path.join(save_folder, 'object_{}_grasp.png'.format(obj_idx))
    vis.capture_screen_image(filename, do_render=True)
    if show:
        o3d.visualization.draw_geometries([model, *grippers])

if __name__ == '__main__':
    camera = 'kinect'
    dataset_root = '../'
    scene_name = 'scene_0000'
    anno_idx = 0
    visAnno(dataset_root, scene_name, anno_idx, camera, num_grasp=1, th=0.5, align_to_table=True, max_width=0.08, save_folder='save_fig', show=False)
    vis6D(dataset_root, scene_name, anno_idx, camera, align_to_table=True, save_folder='save_fig', show=False)

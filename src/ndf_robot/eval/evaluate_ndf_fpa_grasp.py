import os, os.path as osp
import random
import numpy as np
import time
import signal
import torch
import argparse
import shutil

import pybullet as p
import trimesh

from airobot import Robot
from airobot import log_info, log_warn, log_debug, log_critical, set_log_level
from airobot.utils import common
from airobot import log_info
from airobot.utils.common import euler2quat

import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from ndf_robot.utils import util, trimesh_util
from ndf_robot.utils.util import PoseStamped, np2img

from ndf_robot.opt.optimizer import OccNetOptimizer
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
from ndf_robot.config.default_obj_cfg import get_obj_cfg_defaults
from ndf_robot.utils import path_util
from ndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
from ndf_robot.utils.franka_ik import FrankaIK
from ndf_robot.utils.util import PoseStamped, np2img
from ndf_robot.utils.eval_gen_utils import (
    soft_grasp_close, constraint_grasp_close, constraint_obj_world, constraint_grasp_open,
    safeCollisionFilterPair, object_is_still_grasped, get_ee_offset, post_process_grasp_point,
    process_demo_data_rack, process_demo_data_shelf, process_xq_data, process_xq_rs_data, safeRemoveConstraint,
)

placement_surface = ''

def get_pointcloud(obj_id, table_id, rack_link_id, cams):
    # get object (mug) point cloud in target scene
    depth_imgs = []
    seg_idxs = []
    obj_pcd_pts = []
    table_pcd_pts = []
    rack_pcd_pts = []

    obj_pose_world = p.getBasePositionAndOrientation(obj_id)
    obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))
    for i, cam in enumerate(cams.cams):
        # get image and raw point cloud
        rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
        pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0.0, depth_max=np.inf)

        # flatten and find corresponding pixels in segmentation mask
        flat_seg = seg.flatten()
        flat_depth = depth.flatten()
        obj_inds = np.where(flat_seg == obj_id)
        table_inds = np.where(flat_seg == table_id)
        seg_depth = flat_depth[obj_inds[0]]

        obj_pts = pts_raw[obj_inds[0], :]
        obj_pcd_pts.append(util.crop_pcd(obj_pts)) # confirm what the limits mean
        table_pts = pts_raw[table_inds[0], :][::int(table_inds[0].shape[0]/500)]
        table_pcd_pts.append(table_pts)

        if rack_link_id is not None:
            rack_val = table_id + ((rack_link_id+1) << 24)
            rack_inds = np.where(flat_seg == rack_val)
            if rack_inds[0].shape[0] > 0:
                rack_pts = pts_raw[rack_inds[0], :]
                rack_pcd_pts.append(rack_pts)

        depth_imgs.append(seg_depth)
        seg_idxs.append(obj_inds)

    target_obj_pcd_obs = np.concatenate(obj_pcd_pts, axis=0)  # object shape point cloud
    return target_obj_pcd_obs


def get_rack_points(table_id, rack_link_id):
    # Sample points from rack with mug in target pose
    # Step 1 - uniform sampled points
    rack_mesh_file = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/simple_rack.obj')
    rack_mesh = trimesh.load_mesh(rack_mesh_file)
    rack_pts_gt = rack_mesh.sample(500)
    rack_pts_bb = rack_mesh.bounding_box_oriented
    rack_pts_uniform = rack_pts_bb.sample_volume(500)

    rack_pose_world = np.concatenate(p.getLinkState(table_id, rack_link_id)[:2]).tolist()

    uniform_rack_pcd = trimesh.PointCloud(rack_pts_uniform)
    uniform_rack_pose_mat = util.matrix_from_pose(util.list2pose_stamped(rack_pose_world))
    uniform_rack_pcd.apply_transform(uniform_rack_pose_mat)  # points used to represent the rack in demo pose
    rack_optimizer_pts_no_demo = np.asarray(uniform_rack_pcd.vertices)

    # Step 2 - real shape points
    rack_pts_rs = rack_pts_gt  # points used to represent the rack in canonical pose
    rack_pcd_rs = trimesh.PointCloud(rack_pts_rs) #convert np to pointcloud object
    rack_pose_mat = util.matrix_from_pose(util.list2pose_stamped(rack_pose_world)) #get the pose of the table wrt world frame
    rack_pcd_rs.apply_transform(rack_pose_mat)  # points used to represent the rack in demo pose
    rack_optimizer_pts_no_demo_rs = np.asarray(rack_pcd_rs.vertices) #this is to get the vertices of the rigid object
    
    return rack_optimizer_pts_no_demo, rack_optimizer_pts_no_demo_rs


def get_shelf_points(table_id, shelf_link_id):
    # Sample points from shelf with object in target pose
    # Step 1 - uniform sampled points
    shelf_mesh_file = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/shelf_back.stl')
    shelf_mesh = trimesh.load_mesh(shelf_mesh_file)
    shelf_pts_gt = shelf_mesh.sample(500)
    shelf_mesh_bb = shelf_mesh.bounding_box_oriented
    shelf_pts_uniform = shelf_mesh_bb.sample_volume(500)

    shelf_pose_world = np.concatenate(p.getLinkState(table_id, shelf_link_id)[:2]).tolist()

    uniform_shelf_pts = shelf_pts_uniform
    uniform_shelf_pcd = trimesh.PointCloud(uniform_shelf_pts)
    uniform_shelf_pose_mat = util.matrix_from_pose(util.list2pose_stamped(shelf_pose_world))
    uniform_shelf_pcd.apply_transform(uniform_shelf_pose_mat)  # points used to represent the rack in demo pose
    uniform_shelf_pts = np.asarray(uniform_shelf_pcd.vertices)

    # Step 2 - real shape points
    shelf_pts_rs = shelf_pts_gt  # points used to represent the shelf in canonical pose
    shelf_pcd_rs = trimesh.PointCloud(shelf_pts_rs)
    shelf_pose_mat = util.matrix_from_pose(util.list2pose_stamped(shelf_pose_world))
    shelf_pcd_rs.apply_transform(shelf_pose_mat)  # points used to represent the shelf in demo pose
    shelf_pts_rs = np.asarray(shelf_pcd_rs.vertices)

    return uniform_shelf_pts, shelf_pts_rs

def main(args, global_dict):
    # Initialization
    if args.debug:
        set_log_level('debug')
    else:
        set_log_level('info')

    robot = Robot('franka', pb_cfg={'gui': args.pybullet_viz}, arm_cfg={'self_collision': False, 'seed': args.seed})
    ik_helper = FrankaIK(gui=False)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # general experiment + environment setup/scene generation configs
    cfg = get_eval_cfg_defaults()
    config_fname = osp.join(path_util.get_ndf_config(), 'eval_cfgs', args.config + '.yaml')
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        log_info('Config file %s does not exist, using defaults' % config_fname)
    cfg.freeze()

    # object specific configs
    obj_cfg = get_obj_cfg_defaults()
    obj_config_name = osp.join(path_util.get_ndf_config(), args.object_class + '_obj_cfg.yaml')
    obj_cfg.merge_from_file(obj_config_name)
    obj_cfg.freeze()

    shapenet_obj_dir = global_dict['shapenet_obj_dir']
    obj_class = global_dict['object_class']
    eval_save_dir = global_dict['eval_save_dir']

    eval_grasp_imgs_dir = osp.join(eval_save_dir, 'grasp_imgs')
    eval_teleport_imgs_dir = osp.join(eval_save_dir, 'teleport_imgs')
    util.safe_makedirs(eval_grasp_imgs_dir)
    util.safe_makedirs(eval_teleport_imgs_dir)

    test_shapenet_ids = np.loadtxt(osp.join(path_util.get_ndf_share(), '%s_test_object_split.txt' % obj_class), dtype=str).tolist()
    if obj_class == 'mug':
        print("Bowl")
        avoid_shapenet_ids = bad_shapenet_mug_ids_list + cfg.MUG.AVOID_SHAPENET_IDS
    elif obj_class == 'bowl':
        avoid_shapenet_ids = bad_shapenet_bowls_ids_list + cfg.BOWL.AVOID_SHAPENET_IDS
    elif obj_class == 'bottle':
        avoid_shapenet_ids = bad_shapenet_bottles_ids_list + cfg.BOTTLE.AVOID_SHAPENET_IDS
    else:
        test_shapenet_ids = []

    finger_joint_id = 9
    left_pad_id = 9
    right_pad_id = 10
    p.changeDynamics(robot.arm.robot_id, left_pad_id, lateralFriction=1.0)
    p.changeDynamics(robot.arm.robot_id, right_pad_id, lateralFriction=1.0)

    x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
    y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
    table_z = cfg.TABLE_Z

    preplace_horizontal_tf_list = cfg.PREPLACE_HORIZONTAL_OFFSET_TF
    preplace_horizontal_tf = util.list2pose_stamped(cfg.PREPLACE_HORIZONTAL_OFFSET_TF)
    preplace_offset_tf = util.list2pose_stamped(cfg.PREPLACE_OFFSET_TF)

    # Initialize the models
    if args.dgcnn:
        model = vnn_occupancy_network.VNNOccNet(
            latent_dim=256,
            model_type='dgcnn',
            return_features=True,
            sigmoid=True,
            acts=args.acts).cuda()
    else:
        model = vnn_occupancy_network.VNNOccNet(
            latent_dim=256,
            model_type='pointnet',
            return_features=True,
            sigmoid=True).cuda()

    if not args.random:
        checkpoint_path = global_dict['vnn_checkpoint_path']
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        pass

    if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
        placement_surface = 'shelf'
        load_shelf = True
    else:
        placement_surface = 'rack'
        load_shelf = False

    success_list = []
    place_success_list = []
    place_success_teleport_list = []
    grasp_success_list = []

    demo_shapenet_ids = []

    # get objects that we can use for testing
    test_object_ids = []
    shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(shapenet_obj_dir)] if obj_class == 'mug' else os.listdir(shapenet_obj_dir)
    for s_id in shapenet_id_list:
        valid = s_id not in demo_shapenet_ids and s_id not in avoid_shapenet_ids
        if args.only_test_ids:
            valid = valid and (s_id in test_shapenet_ids)

        if valid:
            test_object_ids.append(s_id)

    if args.single_instance:
        test_object_ids = [demo_shapenet_ids[0]]

    # reset scene
    robot.arm.reset(force_reset=True)
    robot.cam.setup_camera(
        focus_pt=[0.4, 0.0, table_z],
        dist=0.9,
        yaw=45,
        pitch=-25,
        roll=0)

    cams = MultiCams(cfg.CAMERA, robot.pb_client, n_cams=cfg.N_CAMERAS)
    cam_info = {}
    cam_info['pose_world'] = []
    for cam in cams.cams:
        cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))

    # put table at right spot
    table_ori = euler2quat([0, 0, np.pi / 2])

    # this is the URDF that was used in the demos -- make sure we load an identical one
    if obj_class == 'mug':
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
    elif obj_class == 'bowl':
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_shelf.urdf')
    else:
        tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_shelf.urdf')

    # print("Table URDF file location: ",tmp_urdf_fname)
    table_id = robot.pb_client.load_urdf(tmp_urdf_fname,
                            cfg.TABLE_POS,
                            table_ori,
                            scaling=cfg.TABLE_SCALING)
    if obj_class == 'mug':
        rack_link_id = 0
        placement_surface = 'rack'
        shelf_link_id = 1
    elif obj_class in ['bowl', 'bottle']:
        print("Bowl!")
        placement_surface = 'shelf'
        rack_link_id = None
        shelf_link_id = 0

    if placement_surface == 'shelf':
        placement_link_id = shelf_link_id
    else:
        placement_link_id = rack_link_id

    def hide_link(obj_id, link_id):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=[0, 0, 0, 0])

    def show_link(obj_id, link_id, color):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=color)

    viz_data_list = []
    for iteration in range(args.start_iteration, args.num_iterations):
        # load a test object
        obj_shapenet_id = random.sample(test_object_ids, 1)[0]
        id_str = 'Shapenet ID: %s' % obj_shapenet_id
        log_info(id_str)

        viz_dict = {}  # will hold information that's useful for post-run visualizations
        eval_iter_dir = osp.join(eval_save_dir, 'trial_%d' % iteration)
        util.safe_makedirs(eval_iter_dir)

        if obj_class in ['bottle', 'jar', 'bowl', 'mug']:
            upright_orientation = common.euler2quat([np.pi/2, 0, 0]).tolist()
        else:
            upright_orientation = common.euler2quat([0, 0, 0]).tolist()

        # for testing, use the "normalized" object
        obj_obj_file = osp.join(shapenet_obj_dir, obj_shapenet_id, 'models/model_normalized.obj')
        obj_obj_file_dec = obj_obj_file.split('.obj')[0] + '_dec.obj'

        scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
        scale_default = cfg.MESH_SCALE_DEFAULT
        if args.rand_mesh_scale:
            mesh_scale = [np.random.random() * (scale_high - scale_low) + scale_low] * 3
        else:
            mesh_scale=[scale_default] * 3

        if args.any_pose:
            if obj_class in ['bowl', 'bottle']:
                rp = np.random.rand(2) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rp[0], rp[1], 0]).tolist()
            else:
                rpy = np.random.rand(3) * (2 * np.pi / 3) - (np.pi / 3)
                ori = common.euler2quat([rpy[0], rpy[1], rpy[2]]).tolist()

            pos = [
                np.random.random() * (x_high - x_low) + x_low,
                np.random.random() * (y_high - y_low) + y_low,
                table_z]
            pose = pos + ori
            rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
            pose_w_yaw = util.transform_pose(util.list2pose_stamped(pose), util.pose_from_matrix(rand_yaw_T))
            pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]
        else:
            pos = [np.random.random() * (x_high - x_low) + x_low, np.random.random() * (y_high - y_low) + y_low, table_z]
            pose = util.list2pose_stamped(pos + upright_orientation)
            rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
            pose_w_yaw = util.transform_pose(pose, util.pose_from_matrix(rand_yaw_T))
            pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]

        viz_dict['shapenet_id'] = obj_shapenet_id
        viz_dict['obj_obj_file'] = obj_obj_file
        if 'normalized' not in shapenet_obj_dir:
            viz_dict['obj_obj_norm_file'] = osp.join(shapenet_obj_dir + '_normalized', obj_shapenet_id, 'models/model_normalized.obj')
        else:
            viz_dict['obj_obj_norm_file'] = osp.join(shapenet_obj_dir, obj_shapenet_id, 'models/model_normalized.obj')
        viz_dict['obj_obj_file_dec'] = obj_obj_file_dec
        viz_dict['mesh_scale'] = mesh_scale

        # convert mesh with vhacd
        if not osp.exists(obj_obj_file_dec):
            p.vhacd(
                obj_obj_file,
                obj_obj_file_dec,
                'log.txt',
                concavity=0.0025,
                alpha=0.04,
                beta=0.05,
                gamma=0.00125,
                minVolumePerCH=0.0001,
                resolution=1000000,
                depth=20,
                planeDownsampling=4,
                convexhullDownsampling=4,
                pca=0,
                mode=0,
                convexhullApproximation=1
            )

        robot.arm.go_home(ignore_physics=True)
        robot.arm.move_ee_xyz([0, 0, 0.2])

        if args.any_pose:
            robot.pb_client.set_step_sim(True)
        if obj_class in ['bowl']:
            robot.pb_client.set_step_sim(True)

        obj_id = robot.pb_client.load_geom(
            'mesh',
            mass=0.01,
            mesh_scale=mesh_scale,
            visualfile=obj_obj_file_dec,
            collifile=obj_obj_file_dec,
            base_pos=pos,
            base_ori=ori)
        p.changeDynamics(obj_id, -1, lateralFriction=0.5)

        if obj_class == 'bowl':
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1, linkIndexB=rack_link_id, enableCollision=False)
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1, linkIndexB=shelf_link_id, enableCollision=False)
            robot.pb_client.set_step_sim(False)

        o_cid = None
        if args.any_pose:
            o_cid = constraint_obj_world(obj_id, pos, ori)
            robot.pb_client.set_step_sim(False)
        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
        time.sleep(1.5)

        hide_link(table_id, rack_link_id)

        if obj_class == 'mug':
            rack_color = p.getVisualShapeData(table_id)[rack_link_id][7]
            show_link(table_id, rack_link_id, rack_color)

        if obj_class == 'bowl':
            for i in range(p.getNumJoints(robot.arm.robot_id)):
                safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id, linkIndexA=i, linkIndexB=rack_link_id, enableCollision=False)
                safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id, linkIndexA=i, linkIndexB=shelf_link_id, enableCollision=False)
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1, linkIndexB=rack_link_id, enableCollision=False)
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1, linkIndexB=shelf_link_id, enableCollision=False)

        ##################################### SHRUTHI ############################################

        # Start with hardcoded target pose
        if obj_class=='mug':
            obj_end_pose = PoseStamped()
            obj_end_pose.pose.position.x = 0.5168864236237299
            obj_end_pose.pose.position.y = 0.1759646156040694
            obj_end_pose.pose.position.z = 1.1698229785998042
            obj_end_pose.pose.orientation.x = 0.29852462500313637
            obj_end_pose.pose.orientation.y = -0.5982721882985429
            obj_end_pose.pose.orientation.z = -0.27218172698990295
            obj_end_pose.pose.orientation.w = 0.6920047286456962
            obj_end_pose_list = util.pose_stamped2list(obj_end_pose)
            viz_dict['final_obj_pose'] = obj_end_pose_list

        elif obj_class in ['bottle', 'bowl']:
            obj_end_pose = PoseStamped()
            obj_end_pose.pose.position.x = 0.37288
            obj_end_pose.pose.position.y = -0.368886
            obj_end_pose.pose.position.z = 1.13
            obj_end_pose.pose.orientation.x = 0.700088958933695
            obj_end_pose.pose.orientation.y = 0.0993752966241366
            obj_end_pose.pose.orientation.z = 0.0993752966241366
            obj_end_pose.pose.orientation.w = 0.700088958933695
            obj_end_pose_list = util.pose_stamped2list(obj_end_pose)
            viz_dict['final_obj_pose'] = obj_end_pose_list

        # Reset scene to Target Object pose scene:
        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=False)
        safeCollisionFilterPair(obj_id, table_id, -1, placement_link_id, enableCollision=False)
        robot.pb_client.set_step_sim(True)
        safeRemoveConstraint(o_cid)
        robot.pb_client.reset_body(obj_id, obj_end_pose_list[:3], obj_end_pose_list[3:])

        time.sleep(1.0)

        obj_pose_world_target = p.getBasePositionAndOrientation(obj_id)
        print("Target object pose: ", obj_pose_world_target)

        if placement_surface == 'shelf':
            print('Using shelf points')
            shelf_optimizer_pts_no_demo, shelf_optimizer_pts_rs_no_demo = get_shelf_points(table_id, shelf_link_id)
            place_optimizer_pts = shelf_optimizer_pts_no_demo
            place_optimizer_pts_rs = shelf_optimizer_pts_rs_no_demo
        else:
            print('Using rack points') #2000x3 shape
            rack_optimizer_pts_no_demo, rack_optimizer_pts_no_demo_rs = get_rack_points(table_id, rack_link_id)
            place_optimizer_pts = rack_optimizer_pts_no_demo
            place_optimizer_pts_rs = rack_optimizer_pts_no_demo_rs

        # We only have a place optimizer - this encodes the rack pcd points w.r.t important geometric features of mug
        # We have no notion of gripper. This is just a snapshot of how the object should be placed
        new_place_optimizer = OccNetOptimizer(
        model,
        query_pts=place_optimizer_pts,
        query_pts_real_shape=place_optimizer_pts_rs,
        opt_iterations=args.opt_iterations)

        # Get point cloud of mug in target configuration 
        if placement_surface == 'shelf':
            target_obj_pcd_obs = get_pointcloud(obj_id, table_id, shelf_link_id, cams)
        else:
            target_obj_pcd_obs = get_pointcloud(obj_id, table_id, rack_link_id, cams)

        # Respawn the sim object back to a random position in source
        teleport_rgb = robot.cam.get_images(get_rgb=True)[0]
        teleport_img_fname = osp.join(eval_teleport_imgs_dir, '%d.png' % iteration)
        np2img(teleport_rgb.astype(np.uint8), teleport_img_fname)
        safeCollisionFilterPair(obj_id, table_id, -1, placement_link_id, enableCollision=True)
        robot.pb_client.set_step_sim(False)
        time.sleep(1.0)

        obj_surf_contacts = p.getContactPoints(obj_id, table_id, -1, placement_link_id)
        touching_surf = len(obj_surf_contacts) > 0
        place_success_teleport = touching_surf
        place_success_teleport_list.append(place_success_teleport)

        time.sleep(1.0)

        robot.pb_client.set_step_sim(False)
        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
        if args.any_pose:
            robot.pb_client.set_step_sim(True)
        safeRemoveConstraint(o_cid)
        p.resetBasePositionAndOrientation(obj_id, pos, ori)
        print("Source object pose: ", p.getBasePositionAndOrientation(obj_id))
        time.sleep(0.5)

        # Get point cloud of mug in source configuration
        if placement_surface == 'shelf':
            source_obj_pcd_obs = get_pointcloud(obj_id, table_id, shelf_link_id, cams)
        else:
            source_obj_pcd_obs = get_pointcloud(obj_id, table_id, rack_link_id, cams)
        source_pts_mean = np.mean(source_obj_pcd_obs, axis=0)
        inliers = np.where(np.linalg.norm(source_obj_pcd_obs - source_pts_mean, 2, 1) < 0.15)[0]
        source_obj_pcd_obs = source_obj_pcd_obs[inliers]

        if obj_class == 'mug':
            rack_color = p.getVisualShapeData(table_id)[rack_link_id][7]
            show_link(table_id, rack_link_id, rack_color)

        # Set demo info: For the NDF optimizer, we need to give an information of these two items
        rack_target_info = {
        'demo_query_pts':place_optimizer_pts,
        'demo_obj_pts':target_obj_pcd_obs,
        }

        single_target = []
        single_target.append(rack_target_info)

        # demo_shapenet_ids.append(shapenet_id)
        new_place_optimizer.set_demo_info(single_target)

        # optimize placement pose [same as with NDF]. Here: query points are pcd of object in source pose
        # rack_relative_pose is the transform which will bring source mug to target pose
        rack_pose_mats, best_rack_idx = new_place_optimizer.optimize_transform_implicit(source_obj_pcd_obs, ee=False)
        rack_relative_pose = util.pose_stamped2list(util.pose_from_matrix(rack_pose_mats[best_rack_idx]))

        # Now transform obj_pose to obj_end_pose using rack_relative_pose
        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))

        obj_start_pose = obj_pose_world

        # Reset scene to Target Object pose scene again - now we find the grasp point:
        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=False)
        safeCollisionFilterPair(obj_id, table_id, -1, placement_link_id, enableCollision=False)
        robot.pb_client.set_step_sim(True)
        safeRemoveConstraint(o_cid)
        robot.pb_client.reset_body(obj_id, obj_end_pose_list[:3], obj_end_pose_list[3:])

        time.sleep(1.0)

        obj_pose_world_target = p.getBasePositionAndOrientation(obj_id)
        print("Target object pose: ", obj_pose_world_target)

        # Get grasp point in target pose of object which is approximately the mug rim
        # For now position is just a hard-coded grasp point which is offset by 0.05 in Y and 0.075 in Z
        # Orientation is for now hard-coded to be a 45 degree gripper orientation about Z.
        # Both position & orientation should come from VGN
        if (obj_class == 'mug'):
            ee_end_pose = [0,0,0,0,0,0,0]
            ee_end_pose[0] = obj_pose_world_target[0][0]
            ee_end_pose[1] = obj_pose_world_target[0][1] - 0.05
            ee_end_pose[2] = obj_pose_world_target[0][2] + 0.07
            ee_end_pose[3] = np.pi
            ee_end_pose[4] = 0
            ee_end_pose[5] = 0.3827
            ee_end_pose[6] = 0.9238

        elif(obj_class == 'bowl'):
            ee_end_pose = [0,0,0,0,0,0,0]
            ee_end_pose[0] = obj_pose_world_target[0][0]
            ee_end_pose[1] = obj_pose_world_target[0][1] - 0.03
            ee_end_pose[2] = obj_pose_world_target[0][2]
            ee_end_pose[3] = np.pi #1.29931294
            ee_end_pose[4] = 0
            ee_end_pose[5] = 0.5
            ee_end_pose[6] = 0
            # ee_end_pose[3] = np.pi
            # ee_end_pose[4] = 0
            # ee_end_pose[5] = 0.3827
            # ee_end_pose[6] = 0.9238
        else:
            raise NotImplementedError


        # Transform estimated grasp to source scene (it will be an inverse transform)
        ee_end_pose = util.list2pose_stamped(ee_end_pose)
        pre_grasp_ee_pose = util.inverse_transform_pose(pose_target=ee_end_pose, pose_transform=util.list2pose_stamped(rack_relative_pose))
        pre_grasp_ee_pose = util.pose_stamped2list(pre_grasp_ee_pose)
        pregrasp_offset_tf = get_ee_offset(ee_pose=pre_grasp_ee_pose)
        pre_pre_grasp_ee_pose = util.pose_stamped2list(
            util.transform_pose(pose_source=util.list2pose_stamped(pre_grasp_ee_pose), pose_transform=util.list2pose_stamped(pregrasp_offset_tf)))

        pre_ee_end_pose2 = util.transform_pose(pose_source=ee_end_pose, pose_transform=preplace_offset_tf)
        pre_ee_end_pose1 = util.transform_pose(pose_source=pre_ee_end_pose2, pose_transform=preplace_horizontal_tf)

        ee_end_pose_list = util.pose_stamped2list(ee_end_pose)
        pre_ee_end_pose1_list = util.pose_stamped2list(pre_ee_end_pose1)
        pre_ee_end_pose2_list = util.pose_stamped2list(pre_ee_end_pose2)

        ##################################### SHRUTHI ############################################

        # attempt grasp and solve for plan to execute placement with arm
        jnt_pos = grasp_jnt_pos = grasp_plan = None
        place_success = grasp_success = False
        for g_idx in range(2):

            # reset everything
            robot.pb_client.set_step_sim(False)
            safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
            if args.any_pose:
                robot.pb_client.set_step_sim(True)
            safeRemoveConstraint(o_cid)
            p.resetBasePositionAndOrientation(obj_id, pos, ori)
            print(p.getBasePositionAndOrientation(obj_id))
            time.sleep(0.5)

            if args.any_pose:
                o_cid = constraint_obj_world(obj_id, pos, ori)
                robot.pb_client.set_step_sim(False)
            robot.arm.go_home(ignore_physics=True)

            # turn OFF collisions between robot and object / table, and move to pre-grasp pose
            for i in range(p.getNumJoints(robot.arm.robot_id)):
                safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())
                safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())
            robot.arm.eetool.open()

            if jnt_pos is None or grasp_jnt_pos is None:
                jnt_pos = ik_helper.get_feasible_ik(pre_pre_grasp_ee_pose)
                grasp_jnt_pos = ik_helper.get_feasible_ik(pre_grasp_ee_pose)

                if jnt_pos is None or grasp_jnt_pos is None:
                    jnt_pos = ik_helper.get_ik(pre_pre_grasp_ee_pose)
                    grasp_jnt_pos = ik_helper.get_ik(pre_grasp_ee_pose)

                    if jnt_pos is None or grasp_jnt_pos is None:
                        jnt_pos = robot.arm.compute_ik(pre_pre_grasp_ee_pose[:3], pre_pre_grasp_ee_pose[3:])
                        grasp_jnt_pos = robot.arm.compute_ik(pre_grasp_ee_pose[:3], pre_grasp_ee_pose[3:])  # this is the pose that's at the grasp, where we just need to close the fingers

            if grasp_jnt_pos is not None and jnt_pos is not None:
                if g_idx == 0:
                    robot.pb_client.set_step_sim(True)
                    robot.arm.set_jpos(grasp_jnt_pos, ignore_physics=True)
                    robot.arm.eetool.close(ignore_physics=True)
                    time.sleep(0.2)
                    grasp_rgb = robot.cam.get_images(get_rgb=True)[0]
                    grasp_img_fname = osp.join(eval_grasp_imgs_dir, '%d.png' % iteration)
                    np2img(grasp_rgb.astype(np.uint8), grasp_img_fname)
                    continue

                ########################### planning to pre_pre_grasp and pre_grasp ##########################
                if grasp_plan is None:
                    plan1 = ik_helper.plan_joint_motion(robot.arm.get_jpos(), jnt_pos)
                    plan2 = ik_helper.plan_joint_motion(jnt_pos, grasp_jnt_pos)
                    if plan1 is not None and plan2 is not None:
                        grasp_plan = plan1 + plan2

                        robot.arm.eetool.open()
                        for jnt in plan1:
                            robot.arm.set_jpos(jnt, wait=False)
                            time.sleep(0.025)
                        robot.arm.set_jpos(plan1[-1], wait=True)
                        for jnt in plan2:
                            robot.arm.set_jpos(jnt, wait=False)
                            time.sleep(0.04)
                        robot.arm.set_jpos(grasp_plan[-1], wait=True)

                        # get pose that's straight up
                        offset_pose = util.transform_pose(
                            pose_source=util.list2pose_stamped(np.concatenate(robot.arm.get_ee_pose()[:2]).tolist()),
                            pose_transform=util.list2pose_stamped([0, 0, 0.15, 0, 0, 0, 1])
                        )
                        offset_pose_list = util.pose_stamped2list(offset_pose)
                        offset_jnts = ik_helper.get_feasible_ik(offset_pose_list)

                        # turn ON collisions between robot and object, and close fingers
                        for i in range(p.getNumJoints(robot.arm.robot_id)):
                            safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=True, physicsClientId=robot.pb_client.get_client_id())
                            safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id, linkIndexA=i, linkIndexB=rack_link_id, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())
                            safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id, linkIndexA=i, linkIndexB=shelf_link_id, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())

                        time.sleep(0.8)
                        obj_pos_before_grasp = p.getBasePositionAndOrientation(obj_id)[0]
                        jnt_pos_before_grasp = robot.arm.get_jpos()
                        soft_grasp_close(robot, finger_joint_id, force=50)
                        safeRemoveConstraint(o_cid)
                        time.sleep(0.8)
                        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=False)
                        time.sleep(0.8)

                        if g_idx == 1:
                            grasp_success = object_is_still_grasped(robot, obj_id, right_pad_id, left_pad_id)

                            if grasp_success:
                            # turn OFF collisions between object / table and object / rack, and move to pre-place pose
                                safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
                                robot.arm.eetool.open()
                                p.resetBasePositionAndOrientation(obj_id, obj_pos_before_grasp, ori)
                                soft_grasp_close(robot, finger_joint_id, force=40)
                                robot.arm.set_jpos(jnt_pos_before_grasp, ignore_physics=True)
                                cid = constraint_grasp_close(robot, obj_id)

                        #########################################################################################################

                        if offset_jnts is not None:
                            offset_plan = ik_helper.plan_joint_motion(robot.arm.get_jpos(), offset_jnts)

                            if offset_plan is not None:
                                for jnt in offset_plan:
                                    robot.arm.set_jpos(jnt, wait=False)
                                    time.sleep(0.04)
                                robot.arm.set_jpos(offset_plan[-1], wait=True)

                        # turn OFF collisions between object / table and object / rack, and move to pre-place pose
                        safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=False)
                        safeCollisionFilterPair(obj_id, table_id, -1, rack_link_id, enableCollision=False)
                        safeCollisionFilterPair(obj_id, table_id, -1, shelf_link_id, enableCollision=False)
                        time.sleep(1.0)

        if grasp_success:
            ####################################### get place pose ###########################################

            pre_place_jnt_pos1 = ik_helper.get_feasible_ik(pre_ee_end_pose1_list)
            pre_place_jnt_pos2 = ik_helper.get_feasible_ik(pre_ee_end_pose2_list)
            place_jnt_pos = ik_helper.get_feasible_ik(ee_end_pose_list)

            if place_jnt_pos is not None and pre_place_jnt_pos2 is not None and pre_place_jnt_pos1 is not None:
                plan1 = ik_helper.plan_joint_motion(robot.arm.get_jpos(), pre_place_jnt_pos1)
                plan2 = ik_helper.plan_joint_motion(pre_place_jnt_pos1, pre_place_jnt_pos2)
                plan3 = ik_helper.plan_joint_motion(pre_place_jnt_pos2, place_jnt_pos)

                if plan1 is not None and plan2 is not None and plan3 is not None:
                    place_plan = plan1 + plan2

                    for jnt in place_plan:
                        robot.arm.set_jpos(jnt, wait=False)
                        time.sleep(0.035)
                    robot.arm.set_jpos(place_plan[-1], wait=True)

                ################################################################################################################

                    # turn ON collisions between object and rack, and open fingers
                    safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
                    safeCollisionFilterPair(obj_id, table_id, -1, rack_link_id, enableCollision=True)
                    safeCollisionFilterPair(obj_id, table_id, -1, shelf_link_id, enableCollision=True)

                    for jnt in plan3:
                        robot.arm.set_jpos(jnt, wait=False)
                        time.sleep(0.075)
                    robot.arm.set_jpos(plan3[-1], wait=True)

                    p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
                    constraint_grasp_open(cid)
                    robot.arm.eetool.open()

                    time.sleep(0.2)
                    for i in range(p.getNumJoints(robot.arm.robot_id)):
                        safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())
                    robot.arm.move_ee_xyz([0, 0.075, 0.075])
                    safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=False)
                    time.sleep(4.0)

                    # observe and record outcome
                    obj_surf_contacts = p.getContactPoints(obj_id, table_id, -1, placement_link_id)
                    touching_surf = len(obj_surf_contacts) > 0
                    obj_floor_contacts = p.getContactPoints(obj_id, robot.arm.floor_id, -1, -1)
                    touching_floor = len(obj_floor_contacts) > 0
                    place_success = touching_surf and not touching_floor

        robot.arm.go_home(ignore_physics=True)

        place_success_list.append(place_success)
        grasp_success_list.append(grasp_success)
        log_str = 'Iteration: %d, ' % iteration
        kvs = {}
        kvs['Place Success'] = sum(place_success_list) / float(len(place_success_list))
        kvs['Place [teleport] Success'] = sum(place_success_teleport_list) / float(len(place_success_teleport_list))
        kvs['Grasp Success'] = sum(grasp_success_list) / float(len(grasp_success_list))
        for k, v in kvs.items():
            log_str += '%s: %.3f, ' % (k, v)
        id_str = ', shapenet_id: %s' % obj_shapenet_id
        log_info(log_str + id_str)

        eval_iter_dir = osp.join(eval_save_dir, 'trial_%d' % iteration)
        if not osp.exists(eval_iter_dir):
            os.makedirs(eval_iter_dir)
        sample_fname = osp.join(eval_iter_dir, 'success_rate_eval_implicit.npz')
        np.savez(
            sample_fname,
            obj_shapenet_id=obj_shapenet_id,
            success=success_list,
            grasp_success=grasp_success,
            place_success=place_success,
            place_success_teleport=place_success_teleport,
            grasp_success_list=grasp_success_list,
            place_success_list=place_success_list,
            place_success_teleport_list=place_success_teleport_list,
            start_obj_pose=util.pose_stamped2list(obj_start_pose),
            best_place_obj_pose=obj_end_pose_list,
            # ee_transforms=pre_grasp_ee_pose_mats,
            obj_transforms=rack_pose_mats,
            mesh_file=obj_obj_file,
            distractor_info=None,
            args=args.__dict__,
            global_dict=global_dict,
            cfg=util.cn2dict(cfg),
            obj_cfg=util.cn2dict(obj_cfg)
        )

        robot.pb_client.remove_body(obj_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--eval_data_dir', type=str, default='eval_data')
    parser.add_argument('--demo_exp', type=str, default='grasp_rim_hang_handle_gaussian_precise_w_shelf')
    parser.add_argument('--exp', type=str, default='test_mug_eval')
    parser.add_argument('--object_class', type=str, default='bowl')
    parser.add_argument('--opt_iterations', type=int, default=500)
    parser.add_argument('--num_demo', type=int, default=12, help='number of demos use')
    parser.add_argument('--any_pose', action='store_true')
    parser.add_argument('--num_iterations', type=int, default=5)
    parser.add_argument('--resume_iter', type=int, default=0)
    parser.add_argument('--config', type=str, default='eval_mug_gen')
    parser.add_argument('--model_path', type=str, required=False, default='multi_category_weights')
    parser.add_argument('--save_vis_per_model', action='store_true', default=True)
    parser.add_argument('--noise_scale', type=float, default=0.05)
    parser.add_argument('--noise_decay', type=float, default=0.75)
    parser.add_argument('--pybullet_viz', action='store_true', default=True)
    parser.add_argument('--dgcnn', action='store_true')
    parser.add_argument('--random', action='store_true', help='utilize random weights')
    parser.add_argument('--early_weight', action='store_true', help='utilize early weights')
    parser.add_argument('--late_weight', action='store_true', help='utilize late weights')
    parser.add_argument('--rand_mesh_scale', action='store_true', default=True)
    parser.add_argument('--only_test_ids', action='store_true',default=True)
    parser.add_argument('--all_cat_model', action='store_true', help='True if we want to use a model that was trained on multipl categories')
    parser.add_argument('--n_demos', type=int, default=0, help='if some integer value greater than 0, we will only use that many demonstrations')
    parser.add_argument('--acts', type=str, default='all')
    parser.add_argument('--old_model', action='store_true', help='True if using a model using the old extents centering, else new one uses mean centering + com offset')
    parser.add_argument('--save_all_opt_results', action='store_true', help='If True, then we will save point clouds for all optimization runs, otherwise just save the best one (which we execute)')
    parser.add_argument('--grasp_viz', action='store_true')
    parser.add_argument('--single_instance', action='store_true')
    parser.add_argument('--non_thin_feature', action='store_true')
    parser.add_argument('--grasp_dist_thresh', type=float, default=0.0025)
    parser.add_argument('--start_iteration', type=int, default=0)


    args = parser.parse_args()

    signal.signal(signal.SIGINT, util.signal_handler)

    obj_class = args.object_class
    shapenet_obj_dir = osp.join(path_util.get_ndf_obj_descriptions(), obj_class + '_centered_obj_normalized')

    demo_load_dir = osp.join(path_util.get_ndf_data(), 'demos', obj_class, args.demo_exp)

    expstr = 'exp--' + str(args.exp)
    modelstr = 'model--' + str(args.model_path)
    seedstr = 'seed--' + str(args.seed)
    full_experiment_name = '_'.join([expstr, modelstr, seedstr])
    eval_save_dir = osp.join(path_util.get_ndf_eval_data(), args.eval_data_dir, full_experiment_name)
    util.safe_makedirs(eval_save_dir)

    vnn_model_path = osp.join(path_util.get_ndf_model_weights(), args.model_path + '.pth')

    global_dict = dict(
        shapenet_obj_dir=shapenet_obj_dir,
        demo_load_dir=demo_load_dir,
        eval_save_dir=eval_save_dir,
        object_class=obj_class,
        vnn_checkpoint_path=vnn_model_path
    )

    main(args, global_dict)

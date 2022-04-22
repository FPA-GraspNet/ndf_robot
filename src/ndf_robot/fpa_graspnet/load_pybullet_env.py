import pybullet as p
from airobot import Robot
from airobot import log_info, log_warn, log_debug, log_critical, set_log_level
from airobot.utils import common
from airobot import log_info
from airobot.utils.common import euler2quat
from ndf_robot.utils.franka_ik import FrankaIK
import torch
import random
import numpy as np
from ndf_robot.config.default_eval_cfg import get_eval_cfg_defaults
import os, os.path as osp
from ndf_robot.utils import path_util
from ndf_robot.config.default_obj_cfg import get_obj_cfg_defaults
from ndf_robot.utils import util, trimesh_util
from ndf_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
from ndf_robot.robot.multicam import MultiCams
from ndf_robot.utils.eval_gen_utils import (
    soft_grasp_close, constraint_grasp_close, constraint_obj_world, constraint_grasp_open,
    safeCollisionFilterPair, object_is_still_grasped, get_ee_offset, post_process_grasp_point,
    process_demo_data_rack, process_demo_data_shelf, process_xq_data, process_xq_rs_data, safeRemoveConstraint,
)
import time



robot = Robot('franka', pb_cfg={'gui': True}, arm_cfg={'self_collision': False, 'seed': 20})
ik_helper = FrankaIK(gui=False)
torch.manual_seed(20)
random.seed(20)
np.random.seed(20)
config = 'eval_mug_gen'
demo_shapenet_ids = []
only_test_ids = True
single_instance = True


config_fname = osp.join(path_util.get_ndf_config(), 'eval_cfgs', config + '.yaml')
 # general experiment + environment setup/scene generation configs
cfg = get_eval_cfg_defaults()
config_fname = osp.join(path_util.get_ndf_config(), 'eval_cfgs', config + '.yaml')
if osp.exists(config_fname):
    cfg.merge_from_file(config_fname)
else:
    log_info('Config file %s does not exist, using defaults' % config_fname)
cfg.freeze()

# object specific configs
obj_cfg = get_obj_cfg_defaults()
obj_config_name = osp.join(path_util.get_ndf_config(), "mug" + '_obj_cfg.yaml')
obj_cfg.merge_from_file(obj_config_name)
obj_cfg.freeze()

shapenet_obj_dir = '/home/shruthi/Workspace/ndf_robot/src/ndf_robot/descriptions/objects/mug_centered_obj_normalized'
obj_class = "mug"
eval_save_dir = '/home/shruthi/Workspace/ndf_robot/src/ndf_robot/eval_data/eval_data/exp--test_mug_eval_model--multi_category_weights_seed--0'

eval_grasp_imgs_dir = osp.join(eval_save_dir, 'grasp_imgs')
eval_teleport_imgs_dir = osp.join(eval_save_dir, 'teleport_imgs')
util.safe_makedirs(eval_grasp_imgs_dir)
util.safe_makedirs(eval_teleport_imgs_dir)

test_shapenet_ids = np.loadtxt(osp.join(path_util.get_ndf_share(), '%s_test_object_split.txt' % obj_class), dtype=str).tolist()
if obj_class == 'mug':
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

x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
table_z = cfg.TABLE_Z


preplace_horizontal_tf_list = cfg.PREPLACE_HORIZONTAL_OFFSET_TF
preplace_horizontal_tf = util.list2pose_stamped(cfg.PREPLACE_HORIZONTAL_OFFSET_TF)
preplace_offset_tf = util.list2pose_stamped(cfg.PREPLACE_OFFSET_TF)


if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
    load_shelf = True
else:
    load_shelf = False

# get objects that we can use for testing
test_object_ids = []
shapenet_id_list = [fn.split('_')[0] for fn in os.listdir(shapenet_obj_dir)] if obj_class == 'mug' else os.listdir(shapenet_obj_dir)
for s_id in shapenet_id_list:
    valid = s_id not in demo_shapenet_ids and s_id not in avoid_shapenet_ids
    if only_test_ids:
        valid = valid and (s_id in test_shapenet_ids)

    if valid:
        test_object_ids.append(s_id)

# reset
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
table_urdf ='/home/shruthi/Workspace/ndf_robot/pybullet-planning/models/franka_description/robots/panda_arm_hand.urdf'
# this is the URDF that was used in the demos -- make sure we load an identical one
tmp_urdf_fname = osp.join(path_util.get_ndf_descriptions(), 'hanging/table/table_rack_tmp.urdf')
# open(tmp_urdf_fname, 'w').write(table_urdf)
table_id = robot.pb_client.load_urdf(tmp_urdf_fname,
                        cfg.TABLE_POS,
                        table_ori,
                        scaling=cfg.TABLE_SCALING)

if obj_class == 'mug':
    rack_link_id = 0
    shelf_link_id = 1
elif obj_class in ['bowl', 'bottle']:
    rack_link_id = None
    shelf_link_id = 0


if cfg.DEMOS.PLACEMENT_SURFACE == 'shelf':
    placement_link_id = shelf_link_id
else:
    placement_link_id = rack_link_id

def show_link(obj_id, link_id, color):
        if link_id is not None:
            p.changeVisualShape(obj_id, link_id, rgbaColor=color)

rack_color = p.getVisualShapeData(table_id)[rack_link_id][7]
show_link(table_id, rack_link_id, rack_color)
# for testing, use the "normalized" object
obj_shapenet_id = random.sample(test_object_ids, 1)[0]
obj_obj_file = osp.join(shapenet_obj_dir, obj_shapenet_id, 'models/model_normalized.obj')
obj_obj_file_dec = obj_obj_file.split('.obj')[0] + '_dec.obj'
pos = [np.random.random() * (x_high - x_low) + x_low, np.random.random() * (y_high - y_low) + y_low, table_z]
rp = np.random.rand(2) * (2 * np.pi / 3) - (np.pi / 3)
ori = common.euler2quat([rp[0], rp[1], 0]).tolist()


obj_id = robot.pb_client.load_geom(
            'mesh',
            mass=0.01,
            mesh_scale=0.3,
            visualfile=obj_obj_file_dec,
            collifile=obj_obj_file_dec,
            base_pos=pos,
            base_ori=ori)
p.changeDynamics(obj_id, -1, lateralFriction=0.5)
mug_pose = p.getBasePositionAndOrientation(obj_id)

print("Mug pose is: ", mug_pose)
# turn OFF collisions between robot and object / table, and move to pre-grasp pose
for i in range(p.getNumJoints(robot.arm.robot_id)):
    safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=table_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())
    safeCollisionFilterPair(bodyUniqueIdA=robot.arm.robot_id, bodyUniqueIdB=obj_id, linkIndexA=i, linkIndexB=-1, enableCollision=False, physicsClientId=robot.pb_client.get_client_id())

for i in range(10000):
    # print(i)
    p.stepSimulation()
    time.sleep(1./240.)
    # robot.arm.move_ee_xyz([0, 0, 0.2])
    robot.arm.go_home(ignore_physics=True)

p.disconnect()

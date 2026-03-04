import math
import random

# Simulation
control_dt = 1. / 240.
margin_error = 0.001
gripper_margin_error = 0.0001
joint_margin_error = 0.01
rel_tol = 1e-4
abs_tol = 0.0

# Robots
gripper_goal_position_open_sawyer = 0.2
gripper_goal_position_closed_sawyer = 1.0
arm_movement_force_sawyer = 5 * 240
gripper_movement_force_sawyer = 1000
ee_index_sawyer = 16

gripper_goal_position_open_franka = 0.04
gripper_goal_position_closed_franka = 0.0005
arm_movement_force_franka = 5 * 240
gripper_movement_force_franka = 1000
ee_index_franka = 11

gripper_goal_position_open_ur3 = -42
gripper_goal_position_closed_ur3 = 44
arm_movement_force_ur3 = 5 * 240
gripper_movement_force_ur3 = 1000
ee_index_ur3 = 8 # This is the end-effector joint (flange-tool0)

# The motor joint of the gripper is the one that is controlled to open/close the gripper
# In the URDF file this joint is usually associated with the <tranmission> tag
robotiq_motor_joint = 1
onrobot_rg2_motor_joint = 1 # For the University of Osaka model this corresponds to the finger_joint

# Environment
base_start_position_sawyer = [0.0, 0.0, 0.0]
base_start_orientation_e_sawyer = [0.0, 0.0, math.pi / 2]
joint_start_positions_sawyer = [-0.0304, -2.0563, -1.1631, -0.3829, 1.3152, 0.1496, 1.4462, -0.2288]
base_start_position_franka = [0.0, 0.0, 0.0]
base_start_orientation_e_franka = [0.0, 0.0, math.pi / 2]
joint_start_positions_franka = [0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0, 0.04, 0.04]

base_table_position_ur3 = [-0.4457, 0.2988, 0.73] #pose original + (-0.4457, 0.2988, 0.73)
#base_start_position_ur3 = [0.0, 0.0, 0.73] #pose original elevada +0.73 (altura da mesa) - base do objeto centralizada na origem do sistema
#base_start_position_ur3 = [0.0, 0.0, 0.0]
base_start_orientation_e_ur3 = [0.0, 0.0, math.pi / 2]
joint_start_positions_ur3 = [-math.pi, -math.pi/2, -math.pi/2, -math.pi/2, math.pi/2, math.radians(358.22)]
#joint_start_positions_ur3 = [0, -math.pi/2, 0, -math.pi/2, 0, 0] # home position

ee_start_position_sawyer = [0.0, 0.6, 0.55]
ee_start_orientation_e_sawyer = [0.0, math.pi, -math.pi / 2]
ee_start_position_franka = [0.0, 0.6, 0.55]
ee_start_orientation_e_franka = [0.0, math.pi, -math.pi / 2]
ee_table_position_ur3 = [-0.33334999998433, 0.5973999999832018, 1.0436499999769566] #pose original + 0.73 em z (altura da mesa)
#ee_start_position_ur3 = [0.11235000001566998, 0.2985999999832018, 1.0436499999769566] #pose original + 0.73 em z (altura da mesa)
#ee_start_position_ur3 = [0.11235000001566998, 0.2985999999832018, 0.3136499999769568] #pose correspondente ao robô carregado no chão
ee_start_orientation_e_ur3 = [3.14159265339116, 2.1137534847394844e-10, 0.031066860890601226]

#object_start_position = [0.05, 0.75, 0.1] #sawyer
#object_start_position = [0.05, 0.75, 0] #franka
table_start_position = [0.0, 0.0, 0.0] #CG da mesa na origem do sistema
object_start_position = [0, 0.26, 0.73] #objetos sobre a mesa (0.73 = altura da mesa)
#object_start_position = [0, 0.26, 0.8] #ur3 - mustard bottle on the table
#object_start_orientation_e = [0.0, 0.0, random.uniform(-math.pi, math.pi)]
table_start_orientation_e = [0.0, 0.0, math.pi]
object_start_orientation_e = [0.0, 0.0, math.pi/3]

global_scaling = 0.08

# Camera
# fov, aspect, near_plane, far_plane = 60, 1.0, 0.01, 10
fov, aspect, near_plane, far_plane = 60, 1.0, 0.01, 100
image_width = 256
image_height = 256

head_camera_position = [0, 1.12, 0.8]
# head_camera_position = [0, 1.2, 1.2] antes
# head_camera_position = [0.0, 1.2, 0.6]
head_camera_orientation_e = [0, 2.0 / 4.5 * math.pi, -math.pi/2]
# head_camera_orientation_e = [0, 2.5 / 4.5 * math.pi, -math.pi/2] antes
# head_camera_orientation_e = [0.0, 3 / 4.5 * math.pi, -math.pi / 2]
wrist_camera_position = [0, 0.26, 1.3]
wrist_camera_orientation_e = [0, -math.pi, -math.pi/2]
# wrist_camera_position = [0, -0.9, 1.6]
# wrist_camera_orientation_e = [0, -3.5 / 4.5 * math.pi, -math.pi/2]

camera_distance = 1.0
#camera_distance = 0.8
camera_yaw = 180.0 #orientação em torno da mesa (eixo z). 360 fica exatamente atras.
#camera_yaw = 225.0
camera_pitch = -30.0 #ângulo de inclinação da câmera. -90 é de cima para baixo. 0 é horizontal.
camera_target_position = [0, 0.26, 0.73]
#camera_target_position = [0.0, 0.6, 0.3]
wrist_camera_offset_sawyer = 0.125

# Object grasping
point_cloud_top_surface_filter = 0.1 #0.1 para garrafa e 0.08 p mustard bottle
#point_cloud_top_surface_filter = 0.06
#bounding_cube_depth_offset = 0.03
bounding_cube_depth_offset = 0.06
gripper_depth_offset_franka = 0.06
gripper_depth_offset_sawyer = -0.12
gripper_depth_offset_ur3 = -0.174 # See https://onrobot.com/storage/datasheets/rg2.pdf

# Segmentation
#segmentation_threshold = 0.5
segmentation_threshold = 0.2

# XMem configuration
xmem_config = {
    "top_k": 30,
    "mem_every": 5,
    "deep_update_every": -1,
    "enable_long_term": True,
    "enable_long_term_count_usage": True,
    "num_prototypes": 128,
    "min_mid_term_frames": 5,
    "max_mid_term_frames": 10,
    "max_long_term_elements": 10000,
}

xmem_visualise_every = 1
xmem_output_every = 1
xmem_lm_input_every = 20

# Multiprocessing
CAPTURE_IMAGES = 1
ADD_BOUNDING_CUBES = 2
ADD_TRAJECTORY_POINTS = 3
EXECUTE_TRAJECTORY = 4
OPEN_GRIPPER = 5
CLOSE_GRIPPER = 6
TASK_COMPLETED = 7
RESET_ENVIRONMENT = 8

# Paths
rgb_image_wrist_path = "./images/rgb_image_wrist.png"
depth_image_wrist_path = "./images/depth_image_wrist.png"
rgb_image_head_path = "./images/rgb_image_head.png"
depth_image_head_path = "./images/depth_image_head.png"
rgb_image_trajectory_path = "./images/trajectory/rgb_image_{step}.png"
depth_image_trajectory_path = "./images/trajectory/depth_image_{step}.png"
bounding_cube_mask_image_path = "./images/bounding_cube_mask_{object}_{mask}.png"
contour_debug_image_path = "./images/contour_debug_{object}_{mask}.png"

langsam_image_path = "./images/langsam_image_{object}.png"
xmem_input_path = "./images/xmem_input.png"
xmem_output_path = "./images/xmem_output_{step}.png"

# Output
OK = "\033[92m"
PROGRESS = "\033[93m"
FAIL = "\033[91m"
ENDC = "\033[0m"

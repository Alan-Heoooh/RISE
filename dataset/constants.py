import numpy as np

TO_TENSOR_KEYS = ['input_coords_list', 'input_feats_list', 'input_force_list', 'input_force_list_normalized' 'action', 'action_normalized']

# camera intrinsics
INTRINSICS = {
    "043322070878": np.array([[909.72656250, 0, 645.75042725, 0],
                              [0, 909.66497803, 349.66162109, 0],
                              [0, 0, 1, 0]]),
    "750612070851": np.array([[922.37457275, 0, 637.55419922, 0],
                              [0, 922.46069336, 368.37557983, 0],
                              [0, 0, 1, 0]]),
    "035622060973": np.array([[918.10614014, 0, 638.70654297,   0.        ],
       [  0.        , 916.94476318, 369.33404541,   0.        ],
       [  0.        ,   0.        ,   1.        ,   0.        ]])
}

# inhand camera serial
INHAND_CAM = ["043322070878"]

# transformation matrix from inhand camera (corresponds to INHAND_CAM[0]) to tcp
INHAND_CAM_TCP = np.array([
    [0, -1, 0, 0],
    [1, 0, 0, 0.077],
    [0, 0, 1, 0.2665],
    [0, 0, 0, 1]
])

# from utils.constants

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])

# tcp normalization and gripper width normalization
TRANS_MIN, TRANS_MAX = np.array([-0.35, -0.35, 0]), np.array([0.35, 0.35, 0.7]) 
MAX_GRIPPER_WIDTH = 0.11 # meter

# force normalization constants
FORCE_MIN = np.array([-80.0, -80.0, -80.0, -10.0, -10.0, -10.0])
FORCE_MAX = np.array([80.0, 80.0, 80.0, 10.0, 10.0, 10.0])

# workspace in camera coordinate
WORKSPACE_MIN = np.array([-0.5, -0.5, 0])
WORKSPACE_MAX = np.array([0.5, 0.5, 1.0])

# safe workspace in base coordinate
SAFE_EPS = 0.002
SAFE_WORKSPACE_MIN = np.array([0.2, -0.4, 0.0])
SAFE_WORKSPACE_MAX = np.array([0.8, 0.4, 0.4])

# gripper threshold (to avoid gripper action too frequently)
GRIPPER_THRESHOLD = 0.02 # meter


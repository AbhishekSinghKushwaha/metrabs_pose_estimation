import os

# Model type for Metrabs
MODEL_TYPE = 'metrabs_eff2l_y4_360'

# Server URL prefix for downloading the model
SERVER_PREFIX = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'

# Cache directory for storing the model
CACHE_DIR = os.path.expanduser(os.path.join('~', '.keras', 'models'))

# SMPL24 joint names for pose estimation
SMPL24_JOINT_NAMES = [
    'Pelvis_0', 'Left_Hip_1', 'Right_Hip_2', 'Spine1_3', 'Left_Knee_4', 'Right_Knee_5',
    'Spine2_6', 'Left_Ankle_7', 'Right_Ankle_8', 'Spine3_9', 'Left_Foot_10', 'Right_Foot_11',
    'Neck_12', 'Left_Shoulder_13', 'Right_Shoulder_14', 'Head_15', 'Left_Upper_Arm_16',
    'Right_Upper_Arm_17', 'Left_Elbow_18', 'Right_Elbow_19', 'Left_Wrist_20', 'Right_Wrist_21',
    'Left_Hand_22', 'Right_Hand_23'
]

# Mapping of SMPL24 joints to Azure Kinect joint indices
JOINT_MAPPING = {
    "Pelvis_0": 0,
    "Left_Hip_1": 18,
    "Right_Hip_2": 22,
    "Spine1_3": 1,
    "Left_Knee_4": 19,
    "Right_Knee_5": 23,
    "Spine2_6": 2,
    "Left_Ankle_7": 20,
    "Right_Ankle_8": 24,
    "Spine3_9": 3,
    "Left_Foot_10": 21,
    "Right_Foot_11": 25,
    "Neck_12": 3,
    "Left_Shoulder_13": 5,
    "Right_Shoulder_14": 12,
    "Head_15": 26,
    "Left_Upper_Arm_16": 5,
    "Right_Upper_Arm_17": 12,
    "Left_Elbow_18": 6,
    "Right_Elbow_19": 13,
    "Left_Wrist_20": 7,
    "Right_Wrist_21": 14,
    "Left_Hand_22": 8,
    "Right_Hand_23": 15
}
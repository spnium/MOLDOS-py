from tools import *
from landmarks.poselandmarks_ import *
from landmarks.handlandmarks_ import *

def check4r(posepos, rhandpos, lhandpos):
    l_hip = posepos[LEFT_HIP]
    r_hip = posepos[RIGHT_HIP]
    l_shoulder = posepos[LEFT_SHOULDER]
    r_shoulder = posepos[RIGHT_SHOULDER]
    l_elbow = posepos[LEFT_ELBOW]
    r_elbow = posepos[RIGHT_ELBOW]
    rhand = rhandpos[MIDDLE_FINGER_MCP]
    
    
    l_angle = (calculate_angle(l_hip, l_shoulder, l_elbow))
    r_angle = calculate_angle(r_hip, r_shoulder, r_elbow)

    return touching(rhand, l_elbow, 60, 60) and l_angle > 140 and r_angle > 140
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
    rhand = rhandpos[PINKY_TIP]
    
    
    l_angle = (calculate_angle(l_hip, l_shoulder, l_elbow))
    r_angle = calculate_angle(r_hip, r_shoulder, r_elbow)

    return touching(rhand, l_elbow, 60, 60) and l_angle > 140 and r_angle > 140

def check4(side, posepos, rhandpos, lhandpos):
    anotherside = 'left' if side == 'right' else 'right'
    
    hip = {'left': posepos[LEFT_HIP], 'right': posepos[RIGHT_HIP]}
    shoulder = {'left': posepos[LEFT_SHOULDER], 'right': posepos[RIGHT_SHOULDER]}
    elbow = {'left': posepos[LEFT_ELBOW], 'right': posepos[RIGHT_ELBOW]}
    hand = {'left': lhandpos[PINKY_TIP], 'right': rhandpos[PINKY_TIP]}
    
    l_angle = calculate_angle(hip["left"], shoulder["left"], elbow["left"])
    r_angle = calculate_angle(hip["right"], shoulder["right"], elbow["right"])
    
    return touching(hand[side], elbow[anotherside], 60, 60)  and l_angle > 140 and r_angle > 140
import numpy as np
from random import randint

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b) # Mid
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def calculateangle(start, mid, end):
    start, mid, end = np.array(start), np.array(mid), np.array(end)
    angle = np.abs(np.arctan2(end[1]-mid[1], end[0]-mid[0]) - np.arctan2(start[1]-mid[1], start[0]-mid[0]))
    angle = (angle * 180.0 / np.pi) % 360.0
    
    if angle > 180:
        angle = 360 - angle
    
    return angle


for _ in range(1_000_000):
    x1y1 = [randint(0, 1280), randint(0, 720)]
    x2y2 = [randint(0, 1280), randint(0, 720)]
    x3y3 = [randint(0, 1280), randint(0, 720)]
    
    mustbetrue = calculateangle(x1y1, x2y2, x3y3) == calculate_angle(x1y1, x2y2, x3y3)
    print(x1y1, x2y2, x3y3, mustbetrue, calculateangle(x1y1, x2y2, x3y3), calculate_angle(x1y1, x2y2, x3y3))
    assert mustbetrue
import numpy as np

aprx = lambda a, b, err=20.0: b + err > a > b - err
translatepos = lambda x: tuple(np.multiply(x, [1280, 720]).astype(int))
touching = lambda a, b, x_err=100, y_err=100: aprx(a[0], b[0], x_err) and aprx(a[1], b[1], y_err)
get_pos = lambda landmarks, landmark: [landmarks[landmark].x, landmarks[landmark].y]

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b) # Mid
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def maprange(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)
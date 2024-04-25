import csv 
import cv2 
import mediapipe as mp 
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
 
def prep_file(file_name):
    num_coords = 75
    landmarks = ['class']
    for val in range(1, num_coords+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    with open(file_name, mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)    
    
def run(image, file_name, class_name):
    image = cv2.imread(image)
    print(image)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=True) as holistic: 
        results = holistic.process(image)
        
        pose = results.pose_landmarks.landmark
        pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
        
        left_hand = results.left_hand_landmarks.landmark
        left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
        
        right_hand = results.right_hand_landmarks.landmark
        right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())
        
        # Concate rows
        row = pose_row + left_hand_row + right_hand_row
        
        # Append class name 
        row.insert(0, class_name)
        
        with open(file_name, mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row) 
        
prep_file('training_data_0.csv')
run("./training_data/pose_04/pong1.jpg", "training_data_0.csv", "pose04_right")
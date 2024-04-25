import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
import numpy as np
import pandas as pd
import pickle

class Detector:
    def __init__(self, holistic, classification_model, flip_pose_horizontally=True) -> None:
        self.holistic = holistic
        self.flip_pose_horizontally = flip_pose_horizontally
        
        with open(classification_model, 'rb') as f:
            self.model = pickle.load(f)
        
    def run(self, image):
        image.flags.writeable = False
        
        if self.flip_pose_horizontally:
            self.results = self.holistic.process(cv2.flip(image, 1))
        else:
            self.results = self.holistic.process(image)

        self.right_hand_landmarks = self.results.right_hand_landmarks
        self.left_hand_landmarks = self.results.left_hand_landmarks    
        self.pose_landmarks = self.results.pose_landmarks
        image.flags.writeable = True
        self.image = image
        
    def detect(self, display_output=True):
        # Export coordinates
        try:
            image = self.image
            results = self.results
            model = self.model
            
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            left_hand = results.left_hand_landmarks.landmark
            left_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in left_hand]).flatten())
            
            right_hand = results.right_hand_landmarks.landmark
            right_hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in right_hand]).flatten())
            
            # Concate rows
            row = pose_row + left_hand_row + right_hand_row

            # Make Detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            print(body_language_class, body_language_prob)
            
            if display_output:
                self.image = cv2.putText(image, f"{body_language_class} Prob:{round(body_language_prob[np.argmax(body_language_prob)],2)}", (100, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA) 
            else:
                self.image = image
        except Exception as e:
            print(e)
       
    def display(self, text, org=(100, 100), font=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255), thickness=2):
        if type(text) == np.array:
            text = np.array2string(text)
        else:
            text = str(text)
        
        org = (int(org[0]), int(org[1]))
         
        self.image = cv2.putText(self.image, text, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA) 
         
    def display_output(self, name):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)   
        cv2.imshow(name, self.image)

    def draw(self):
        self.image = cv2.flip(self.image, 1)
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(self.image, self.results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(self.image, self.results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(self.image, self.results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )
        
        # 4. Pose Detections
        mp_drawing.draw_landmarks(self.image, self.results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        self.image = cv2.flip(self.image, 1)
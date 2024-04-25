import mediapipe as mp
import numpy as np
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.hands import HandLandmark
from landmarks.poselandmarks_ import *
from landmarks.handlandmarks_ import *

translatepos = lambda pos, dimension=(1280, 720): tuple(np.multiply(pos, dimension).astype(int))
approximate = lambda a, b, err=20.0: b + err > a > b - err
touching = lambda a, b, x_err=100, y_err=100: approximate(a[0], b[0], x_err) and approximate(a[1], b[1], y_err)
middle_point = lambda xy1, xy2: ((xy1[0] + xy2[0]) / 2, (xy1[1] + xy2[1]) / 2)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b) # Mid
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

class Detector:
    def __init__(self, holistic_model, flip_pose_horizontally=True) -> None:
        self.holistic = holistic_model
        self.flip_pose_horizontally = flip_pose_horizontally
        
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
        self.get_pose_coordinates()
        self.get_hands_coordinates()
    
    def get_pose_coordinates(self):
        self.pose_coordinates = []
        try:
            for landmark in self.pose_landmarks.landmark:
                if self.flip_pose_horizontally:
                    coords = translatepos((landmark.x, landmark.y))
                    self.pose_coordinates.append((1280 - coords[0], coords[1]))
                else:
                    self.pose_coordinates.append(translatepos((landmark.x, landmark.y)))
        except AttributeError:
            self.pose_coordinates = [None for _ in PoseLandmark]
                
    def get_hands_coordinates(self):
        self.right_hand_coordinates = []
        self.left_hand_coordinates = []
        if self.right_hand_landmarks:
            for landmark in self.right_hand_landmarks.landmark:
                coords = translatepos((landmark.x, landmark.y))
                self.right_hand_coordinates.append((1280 - coords[0], coords[1]))
        else:
            self.right_hand_coordinates = [None for _ in HandLandmark]
        if self.left_hand_landmarks:
            for landmark in self.left_hand_landmarks.landmark:
                coords = translatepos((landmark.x, landmark.y))
                self.right_hand_coordinates.append((1280 - coords[0], coords[1]))
        else:
            self.left_hand_coordinates = [None for _ in HandLandmark]
            
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

class ExerciseDetector(Detector):
    def exercise_4(self, side):
        side = side.lower()
        anotherside = 'left' if side == 'right' else 'right'
        hands = {'left': self.left_hand_coordinates, 'right': self.right_hand_coordinates}
        
        if (not self.pose_coordinates[0]) or (not hands[side][0]):
            return False

        hip = {'left': self.pose_coordinates[LEFT_HIP], 'right': self.pose_coordinates[RIGHT_HIP]}
        shoulder = {'left': self.pose_coordinates[LEFT_SHOULDER], 'right': self.pose_coordinates[RIGHT_SHOULDER]}
        elbow = {'left': self.pose_coordinates[LEFT_ELBOW], 'right': self.pose_coordinates[RIGHT_ELBOW]}
        left_angle = calculate_angle(hip["left"], shoulder["left"], elbow["left"])
        right_angle = calculate_angle(hip["right"], shoulder["right"], elbow["right"])
        return touching(hands[side][PINKY_TIP], elbow[anotherside], 60, 60)  and left_angle > 160 and right_angle > 160

          
if __name__=='__main__':
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    detector = ExerciseDetector(holistic)
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        frame = cv2.flip(cap.read()[1], 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        detector.run(image)
        rh = detector.right_hand_coordinates[MIDDLE_FINGER_MCP]
        if rh:
            detector.display("RIGHT_HAND", detector.right_hand_coordinates[MIDDLE_FINGER_MCP])
        print(detector.exercise_4("right"))
        
        detector.draw()
        detector.display_output("MOLDOS")
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
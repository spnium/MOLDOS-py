import mediapipe as mp
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.hands import HandLandmark
from landmarks.poselandmarks_ import *
from landmarks.handlandmarks_ import *
import numpy as np

approximate = lambda a, b, err=20.0: b + err > a > b - err
translatepos = lambda x: tuple(np.multiply(x, [1280, 720]).astype(int))
touching = lambda a, b, x_err=100, y_err=100: approximate(a[0], b[0], x_err) and approximate(a[1], b[1], y_err)

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b) # Mid
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle
    
class Detector:
    def __init__(self, hands, pose) -> None:
        self.hands = hands
        self.pose = pose
    
    def run(self, image):
        image.flags.writeable = False
        self.hands_results = self.hands.process(image)
        self.pose_results = self.pose.process(cv2.flip(image, 1))
        image.flags.writeable = True
        self.image = image
        self.get_pose_positions()
        self.get_hands_positions()
        
    def get_pose_positions(self):
        lmlist = []
        if self.pose_results.pose_landmarks:
            for lm in self.pose_results.pose_landmarks.landmark:
                cx,cy = translatepos((lm.x, lm.y))
                lmlist.append((1280 - cx, cy))
            self.pose_pos = lmlist
            
        else:
            self.pose_pos = ["N/A" for _ in PoseLandmark]
            
        return self.pose_pos
    
    def get_hands_positions(self):
        hands_results = self.hands_results
        hands_pos = {}
        hand_types = []
        hand_pos = []
        
        if hands_results.multi_hand_landmarks:
            if len(hands_results.multi_handedness) > 2:
                raise Exception("Too many hands detected")
                
            for hand in hands_results.multi_handedness:
                hand_types.append(hand.classification[0].label)
                
            for i, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                hand_pos = []
                for landmark in hand_landmarks.landmark:
                    hand_pos.append(translatepos((landmark.x, landmark.y)))

                hands_pos[hand_types[i]] = hand_pos

        self.lhand_pos = hands_pos.get("Left", ["N/A" for _ in HandLandmark])
        self.rhand_pos = hands_pos.get("Right", ["N/A" for _ in HandLandmark])
        
        return self.lhand_pos, self.rhand_pos
    
    def check_pose(self, pose, side):
        side = side.lower()
        
        posepos = self.pose_pos
        rhandpos = self.rhand_pos
        lhandpos = self.lhand_pos
        anotherside = 'left' if side == 'right' else 'right'
        
        hip = {'left': posepos[LEFT_HIP], 'right': posepos[RIGHT_HIP]}
        shoulder = {'left': posepos[LEFT_SHOULDER], 'right': posepos[RIGHT_SHOULDER]}
        elbow = {'left': posepos[LEFT_ELBOW], 'right': posepos[RIGHT_ELBOW]}
        hand = {'left': lhandpos[PINKY_TIP], 'right': rhandpos[PINKY_TIP]}
        
        pose1 = lambda: False
        pose2 = lambda: False
        pose3 = lambda: False
        pose4 = lambda:touching(hand[side], elbow[anotherside], 60, 60)  and calculate_angle(hip["left"], shoulder["left"], elbow["left"]) > 160 and calculate_angle(hip["right"], shoulder["right"], elbow["right"]) > 160
        pose5 = lambda: False
        poses = [pose1, pose2, pose3, pose4, pose5]
        
        try:
            return poses[pose - 1]()
        except Exception as e:
            return False
    
    def draw(self):
        image = self.image
        hands_results = self.hands_results
        pose_results = self.pose_results
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        image = cv2.flip(image, 1)
        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        image = cv2.flip(image, 1)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                
        cv2.imshow('MOLDOS', image)

if __name__=='__main__':
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
            
    detector = Detector(hands=hands, pose=pose)
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        frame = cv2.flip(cap.read()[1], 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        detector.run(image)
        detector.draw()
        print(detector.check_pose(4, "right"))
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
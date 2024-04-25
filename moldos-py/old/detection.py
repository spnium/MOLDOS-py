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
translatepos = lambda x, dimension=(1280, 720): tuple(np.multiply(x, dimension).astype(int))
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
            self.pose_pos = [None for _ in PoseLandmark]
            
        return self.pose_pos
    
    def get_hands_positions(self):
        hands_results = self.hands_results
        hands_pos = {}
        hand_types = []
        hand_pos = []
        
        if hands_results.multi_hand_landmarks and len(hands_results.multi_handedness) <= 2:
                
            for hand in hands_results.multi_handedness:
                hand_types.append(hand.classification[0].label)
                
            for i, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                hand_pos = []
                for landmark in hand_landmarks.landmark:
                    hand_pos.append(translatepos((landmark.x, landmark.y)))

                hands_pos[hand_types[i]] = hand_pos

        self.lhand_pos = hands_pos.get("Left", [None for _ in HandLandmark])
        self.rhand_pos = hands_pos.get("Right", [None for _ in HandLandmark])
        
        return self.lhand_pos, self.rhand_pos
    
    def check_pose(self, pose, side):
        side = side.lower()
        
        posepos = self.pose_pos
        rhandpos = self.rhand_pos
        lhandpos = self.lhand_pos
        anotherside = 'left' if side == 'right' else 'right'
        
        wrist = {'left':posepos[LEFT_WRIST], 'right':posepos[RIGHT_WRIST]}
        hip = {'left': posepos[LEFT_HIP], 'right': posepos[RIGHT_HIP]}
        shoulder = {'left': posepos[LEFT_SHOULDER], 'right': posepos[RIGHT_SHOULDER]}
        elbow = {'left': posepos[LEFT_ELBOW], 'right': posepos[RIGHT_ELBOW]}
        hands = {'left': lhandpos, 'right': rhandpos}
        eye_inner = {'left': posepos[LEFT_EYE_INNER], 'right': posepos[RIGHT_EYE_INNER]}
        nose = posepos[NOSE]
        
        def pose2():
            middle = middle_point(eye_inner[side], eye_inner[anotherside])
            top_of_head = (middle[0], middle[1] - int(abs(middle[1] - nose[1]) * 3))
            return top_of_head
            
        pose1 = lambda: False
        
        def pose3():
            return False
            # return middle_point(wrist[side], elbow[side])
        
        def pose4():
            left_angle = calculate_angle(hip["left"], shoulder["left"], elbow["left"])
            right_angle = calculate_angle(hip["right"], shoulder["right"], elbow["right"])
            return touching(hands[side][PINKY_TIP], elbow[anotherside], 60, 60)  and left_angle > 160 and right_angle > 160
        
        def pose5():
            return False
            # return touching(hands[side][MIDDLE_FINGER_MCP], hands[anotherside][MIDDLE_FINGER_MCP], 160, 160)
            
        poses = [pose1, pose2, pose3, pose4, pose5]
        
        try:
            return poses[pose - 1]()
        except Exception as e:
            return False
        
class DetectorCV(Detector):
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
        
        self.image = image
        
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

if __name__=='__main__':
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
            
    detector = DetectorCV(hands=hands, pose=pose)
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        frame = cv2.flip(cap.read()[1], 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        detector.run(image)
        pose_pos = detector.pose_pos
        right_hand = detector.rhand_pos
        left_hand = detector.lhand_pos
        
        p5 = detector.check_pose(2, "right")
        detector.display(p5, org=p5)
        
        detector.draw()
        detector.display_output("MOLDOS")
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
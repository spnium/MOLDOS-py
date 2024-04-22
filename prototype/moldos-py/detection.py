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
from tools import *
    
class Detector:
    def __init__(self, hands, pose) -> None:
        self.hands = hands
        self.pose = pose
    
    def run(self, image):
        self.hands_results = self.hands.process(image)
        self.pose_results = self.pose.process(cv2.flip(image, 1))
        self.image = image
        self.__get_pose_pos()
        self.__get_hands_pos()

    def __get_pose_pos(self):
        pose_results = self.pose_results
        pose_pos = []
        
        try:
            poselandmarks = pose_results.pose_landmarks.landmark
        except AttributeError:
            for _ in PoseLandmark:
                pose_pos.append("N/A")
            self.pose_pos = pose_pos
        
        for i in PoseLandmark:
            try:
                # 1280-0 -> 0-1280
                x, y = translatepos(get_pos(poselandmarks, i))
                pose_pos.append((1280 - x, y))

            except Exception as e:
                pose_pos.append("N/A")
        
        self.pose_pos = pose_pos
    
    def __get_hands_pos(self):
        hands_results = self.hands_results
        lhand_pos = []
        rhand_pos = []
        _Hands = []
        handsType = []
        try:
            for hand in hands_results.multi_handedness:
                handType=hand.classification[0].label
                handsType.append(handType)
            for handLandMarks in hands_results.multi_hand_landmarks:
                Hand_ = []
                for landMark in handLandMarks.landmark:
                    Hand_.append(translatepos(((landMark.x), (landMark.y))))
                _Hands.append(Hand_)
            
            try:
                handsType[1]
                rhand_pos = _Hands[1]
                lhand_pos = _Hands[0]
            except:
                if handsType[0] == "Right" and _Hands:
                    rhand_pos = _Hands[0]
                    for _ in HandLandmark:
                        lhand_pos.append("N/A")
                elif handsType[0] == "Left" and _Hands:
                    lhand_pos = _Hands[0]
                    for _ in HandLandmark:
                        rhand_pos.append("N/A")
                else:
                    for _ in HandLandmark:
                        rhand_pos.append("N/A")
                        lhand_pos.append("N/A")
                
        except Exception as e:
            for _ in HandLandmark:
                rhand_pos.append("N/A")
                lhand_pos.append("N/A")
        
        self.rhand_pos = rhand_pos
        self.lhand_pos = lhand_pos
        
    
    def check4(self, side):
        posepos = self.pose_pos
        rhandpos = self.rhand_pos
        lhandpos = self.lhand_pos
        anotherside = 'left' if side == 'right' else 'right'
        
        hip = {'left': posepos[LEFT_HIP], 'right': posepos[RIGHT_HIP]}
        shoulder = {'left': posepos[LEFT_SHOULDER], 'right': posepos[RIGHT_SHOULDER]}
        elbow = {'left': posepos[LEFT_ELBOW], 'right': posepos[RIGHT_ELBOW]}
        hand = {'left': lhandpos[PINKY_TIP], 'right': rhandpos[PINKY_TIP]}
        
        l_angle = calculate_angle(hip["left"], shoulder["left"], elbow["left"])
        r_angle = calculate_angle(hip["right"], shoulder["right"], elbow["right"])
        
        return touching(hand[side], elbow[anotherside], 60, 60)  and l_angle > 140 and r_angle > 140
    
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
        
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
        
detector = Detector(hands=hands, pose=pose)

if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        frame = cv2.flip(cap.read()[1], 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        detector.run(image)
        detector.draw()
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
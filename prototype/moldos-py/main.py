from detection import *
import cv2
import mediapipe as mp

hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)

def set1(detector: DetectorCV):
    detector.run
    pose_pos = detector.pose_pos
    right_hand = detector.rhand_pos
    left_hand = detector.lhand_pos
    

        
detector = DetectorCV(hands=hands, pose=pose)

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
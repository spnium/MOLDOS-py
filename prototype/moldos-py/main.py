from detection import *
import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

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
import mediapipe as mp
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.hands import HandLandmark
from tools import *
import exercises

cap = cv2.VideoCapture(0)

falsebuffer = 0
active = False

with mp_holistic.Holistic(min_detection_confidence=0.45, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        try:
            poselandmarks = results.pose_landmarks.landmark
            try:
                rhandlandmarks = results.right_hand_landmarks.landmark
            except Exception as e:
                # print(e)
                pass
            try:
                lhandlandmarks = results.left_hand_landmarks.landmark
            except Exception as e:
                # print(e)
                pass
            
            pose_pos = []
            lhand_pos = []
            rhand_pos = []
            
            for i in PoseLandmark:
                try:
                    pose_pos.append(translatepos(get_pos(poselandmarks, i)))
                except Exception as e:
                    print(e)
                    pose_pos.append("N/A")
            
            for i in HandLandmark:
                try:
                    rhand_pos.append(translatepos(get_pos(rhandlandmarks, i)))
                    lhand_pos.append(translatepos(get_pos(lhandlandmarks, i)))
                except:
                    rhand_pos.append("N/A")
                    lhand_pos.append("N/A")

            if not active:
                correctpos = False
                color = (255, 0, 0)

            try:
                e4r = exercises.check4r(pose_pos, rhand_pos, lhand_pos)
                active = True
                if e4r:
                    falsebuffer = 0
                    correctpos = True
                    color = (0, 255, 0)
                elif not e4r and falsebuffer <= 5:
                    falsebuffer += 1
                    correctpos = True
                else:
                    correctpos = False
                    color = (255, 0, 0)
                    
            except Exception as e:
                # print(e)
                pass
            
            image = cv2.flip(image, 1)
            image = cv2.putText(image, str(correctpos), (40, 80), cv2.FONT_HERSHEY_SIMPLEX,  
                   2, color, 3, cv2.LINE_AA)
            image = cv2.flip(image, 1) 
            
        except Exception as e:
            # print(e)
            pass
                
        # Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # Pose
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                
        cv2.imshow('Raw Webcam Feed', cv2.flip(image, 1))

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
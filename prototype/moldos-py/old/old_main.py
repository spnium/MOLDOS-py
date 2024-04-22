import mediapipe as mp
import cv2
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.hands import HandLandmark
from tools import *
import exercises

cap = cv2.VideoCapture(0)

falsebuffer = 0
active = False

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    
while cap.isOpened():
    frame = cv2.flip(cap.read()[1], 1)
  
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(image)
    pose_results = pose.process(cv2.flip(image, 1))
    
    try:
        poselandmarks = pose_results.pose_landmarks.landmark

        pose_pos = []
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
                    for i in HandLandmark:
                        lhand_pos.append("N/A")
                elif handsType[0] == "Left" and _Hands:
                    lhand_pos = _Hands[0]
                    for i in HandLandmark:
                        rhand_pos.append("N/A")
                else:
                    for i in HandLandmark:
                        rhand_pos.append("N/A")
                        lhand_pos.append("N/A")
                
        except Exception as e:
            # print(e)
            pass

        for i in PoseLandmark:
            try:
                # 1280-0 -> 0-1280
                x, y = translatepos(get_pos(poselandmarks, i))
                pose_pos.append((1280 - x, y))

            except Exception as e:
                print(e)
                pose_pos.append("N/A")

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

        image = cv2.putText(image, str(correctpos), (40, 80), cv2.FONT_HERSHEY_SIMPLEX,  
                2, color, 3, cv2.LINE_AA)
        
    except Exception as e:
        # print(e)
        pass
            
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

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
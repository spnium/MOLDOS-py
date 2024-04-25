from detector import *

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
detector = DetectorPose(pose, '/Users/maytanan/Desktop/frMOLDOS/training/body_language.pkl', flip_pose_horizontally=True)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    frame = cv2.flip(cap.read()[1], 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    detector.run(image)
    detector.detect()
    
    detector.draw()
    detector.display_output("MOLDOS")
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
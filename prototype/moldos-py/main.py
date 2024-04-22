from detection import *
import cv2

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
import cv2
from ultralytics import YOLO
model =YOLO("runs/detect/train2/weights/best.pt")

cap=cv2.VideoCapture(0)

while True:
    r,frame=cap.read()
    if not r: 
        break

    frame = cv2.flip(frame, 1)
    
    result=model(frame)

    annotated_frame=result[0].plot()
    cv2.imshow("Detection",annotated_frame)

    if cv2.waitKey(1) & 0XFF==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
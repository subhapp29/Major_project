import os 
from kalmanfilter import KalmanFilter
import cv2 
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
cap = cv2.VideoCapture(0)
kf = KalmanFilter()

model = YOLO('../best.pt')
while True:
    ret,frame = cap.read()

    results = model(frame)
    for res in results:
        annotator = Annotator(frame)

        detections = []
        for r in res.boxes.data.tolist():
                x1,y1,x2,y2,score,class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y2 = int(y2)
                y1 = int(y1)
                cx = (x1+(x2-x1)/2)
                cy = (y1+(y2-y1)/2)

                predicted = kf.predict(cx, cy)
                # #cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 4)
                # cv2.circle(frame, (cx, cy), 20, (0, 0, 255), 4)
                # cv2.circle(frame, (predicted[0], predicted[1]), 20, (255, 0, 0), 4)
                # for i in range(0,50):
                #         predicted = kf.predict(cx,cy)
                # cv2.circle(frame, (predicted[0], predicted[1]), 20, (255, 0, 0), 4)

                # cv2.imshow("Frame", frame)



                cv2.circle(frame,(x1,int(cy)),5,(255,0,255),cv2.FILLED)
                cv2.circle(frame,(x2,int(cy)),5,(255,0,255),cv2.FILLED)
                cv2.circle(frame, (predicted[0], predicted[1]), 20, (255, 0, 0), 4)
                predict = []
                for i in range(0,100):
                        predicted = kf.predict(cx,cy)
                        predict.append(predicted)
                for i in range(0,len(predict),2):
                      cv2.line(frame,predict[i],predict[i+1],(255, 0, 0), 4)
                cv2.circle(frame, (predicted[0], predicted[1]), 20, (255, 0, 0), 4)

                w = x2-x1
                W = 6.58
                #             d = 50 
                #             f = (w*d)/W
                f = 660
                d = (f*W)/w
                #             print(d)
                pos1 = [x1,y1,x2,y2]
                pos = [cx,cy,d]
                detections.append([int(x1),int(y1),int(x2),int(y2)])
                #             cv2.rectangle(frame, (x1, y1), (x2, y2),(0,255,0),2)
                cv2.putText(frame, str(pos), (x1+15, y1-15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2 )

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   
cap.release()
cv2.destroyAllWindows()

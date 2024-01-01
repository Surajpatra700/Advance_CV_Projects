import cv2
import time
import numpy as np
import poseMethod as pm

# img = cv2.imread("AI_Training/training.jpg")
cap = cv2.VideoCapture(0)
detector = pm.poseDetector()

count = 0
dir = 0

while True:

    success, img = cap.read()
    img = cv2.resize(img, (1280,720), interpolation=cv2.INTER_AREA)
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        # right arm
        detector.findAngle(img, 12,14,16)
        # # left arm
        angle = detector.findAngle(img, 11,13,15)

        per = np.interp(angle,(60,160),(0,100))
        bar = np.interp(angle,(60,160),(650,100))
        # print(angle,per)

        # Check for no. of Pushups
        color = (255,0,255)
        if per == 100:
            color = (0,255,0)
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            color = (0,255,0)
            if dir == 1:
                count += 0.5
                dir = 0

        # for bar
        cv2.rectangle(img, (1100,100),(1175,658),color,3)
        cv2.rectangle(img, (1100,int(bar)),(1175,658),color,cv2.FILLED)
        cv2.putText(img, f'{int(per)} %', (1100,75),cv2.FONT_HERSHEY_PLAIN, 3,color,3)

        # for pushup count
        cv2.putText(img, f'{count}', (70,50),cv2.FONT_HERSHEY_PLAIN, 3,(255,0,0),3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
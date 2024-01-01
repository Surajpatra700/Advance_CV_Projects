import cv2
import time
import poseMethod as pm

cap = cv2.VideoCapture("./video1.mp4")
pTime = 0
new_width = 640  # Adjust as needed
new_height = 480  # Adjust as needed

detector = pm.poseDetector()

while True:
    success, img = cap.read()
    resized_frame = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    resized_frame = detector.findPose(resized_frame=resized_frame)
    lmList = detector.findPosition(resized_frame=resized_frame)
    print(lmList[10])
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(resized_frame,str(int(fps)),(50,30), cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)

    cv2.imshow("Image",resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
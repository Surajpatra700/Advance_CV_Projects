import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture("./test-video.mp4")
cTime = 0
pTime = 0

new_width = 640  # Adjust as needed
new_height = 480  # Adjust as needed


while True:
    success, img = cap.read()

    resized_frame = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    imgRGB = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(resized_frame,results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = resized_frame.shape
            print([id,lm])
            cx,cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(resized_frame,(cx,cy), 10, (255,0,0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(resized_frame,str(int(fps)),(50,30), cv2.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)

    cv2.imshow("Image",resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
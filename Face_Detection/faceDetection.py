import cv2
import mediapipe as mp
import time

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

cap = cv2.VideoCapture("./video1.mp4")
pTime = 0
while True:
    success, img = cap.read()
    resized_frame = cv2.resize(img, (640,480),interpolation=cv2.INTER_AREA)
    imgRGB = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(resized_frame,detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            h,w,c = resized_frame.shape
            bboxC = detection.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                int(bboxC.width * w), int(bboxC.height * h)
            
            cv2.rectangle(resized_frame,bbox, (255,0,255),2)
            cv2.putText(resized_frame,f'{int(detection.score[0] * 100)}%',(bbox[0], bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(resized_frame,str(int(fps)),(30,80), cv2.FONT_HERSHEY_COMPLEX,2
                 (0,255,0),3)
    cv2.imshow("Image",resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

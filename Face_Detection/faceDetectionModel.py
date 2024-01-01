import cv2
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self,minDetectionCon = 0.5):
        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self,resized_frame,draw=True):
        imgRGB = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                h,w,c = resized_frame.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                    int(bboxC.width * w), int(bboxC.height * h)
                
                bboxs.append([id, bbox, detection.score])

                resized_frame = self.fancyDraw(resized_frame,bbox)
                
                # cv2.rectangle(resized_frame,bbox, (255,0,255),2)
                cv2.putText(resized_frame,f'{int(detection.score[0] * 100)}%',(bbox[0], bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
        return resized_frame, bboxs
    
    def fancyDraw(self, resized_frame, bbox, l=10, t=5):
        x,y,w,h = bbox
        x1,y1 = x+w, y+h
        cv2.rectangle(resized_frame, bbox, (255,0,255), 2)
        cv2.line(resized_frame, (x,y), (x+l, y), (255,0,255),t)
        cv2.line(resized_frame, (x,y), (x, y+l), (255,0,255),t)
        cv2.line(resized_frame, (x1,y1), (x1-l, y1), (255,0,255),t)
        cv2.line(resized_frame, (x1,y1), (x1, y1-l), (255,0,255),t)
        return resized_frame



def main():
    cap = cv2.VideoCapture("./video1.mp4")
    pTime = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        resized_frame = cv2.resize(img, (640,480),interpolation=cv2.INTER_AREA)
        resized_frame, bboxs = detector.findFaces(resized_frame=resized_frame)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(resized_frame,str(int(fps)),(30,80), cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),3)
        cv2.imshow("Image",resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
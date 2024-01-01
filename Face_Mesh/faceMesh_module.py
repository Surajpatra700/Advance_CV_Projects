import cv2
import mediapipe as mp
import time



class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticmethod
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,True,self.minDetectionCon,self.minTrackCon)
        self.drawSpecs = self.mpDraw.DrawingSpec(thickness=0, circle_radius=0)

    def findFaceMesh(self,resized_img,draw=True):

        self.imgRGB = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(image=resized_img,
              landmark_list=faceLms,
              connections=self.mpFaceMesh.FACEMESH_TESSELATION,
              landmark_drawing_spec=self.drawSpecs,
              connection_drawing_spec=self.mp_drawing_styles
              .get_default_face_mesh_tesselation_style())
                
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    h,w,c = resized_img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    # print(id, x, y)
                    face.append([x,y])

                faces.append(face)
        return resized_img, faces



def main():
    pTime = 0
    cap = cv2.VideoCapture("./video1.mp4")
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        resized_img = cv2.resize(img,(640,480),interpolation=cv2.INTER_AREA)
        resized_img, faces = detector.findFaceMesh(resized_img=resized_img)
        if len(faces) != 0:
            print(len(faces))

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(resized_img,f"{int(fps)}", (30,80), cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
        cv2.imshow("Image",resized_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
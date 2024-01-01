import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()
drawSpecs = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

pTime = 0
cap = cv2.VideoCapture("./video1.mp4")

while True:
    success, img = cap.read()
    resized_img = cv2.resize(img,(640,480),interpolation=cv2.INTER_AREA)
    imgRGB = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(image=resized_img,
          landmark_list=faceLms,
          connections=mpFaceMesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
            
            for id, lm in enumerate(faceLms.landmark):
                h,w,c = resized_img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                print(id, x, y)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(resized_img,f"{int(fps)}", (30,80), cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)
    cv2.imshow("Image",resized_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

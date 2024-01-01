# import cv2
# import numpy as np
# import utils

# webCamFeed = True
# pathImage = "doc.jpg"
# cap = cv2.VideoCapture(0)
# cap.set(10,160)
# heightImg = 640
# widthImg = 480

# ######################################

# utils.initializeTrackbars()
# count = 0

# while True:

#     # Blank Image
#     imgBlank = np.zeros((heightImg,widthImg,3),np.uint8) # CREATE A BLANK IMAGE FOR TESTING

#     if webCamFeed: success,img = cap.read()
#     else: img = cv2.imread(pathImage)
#     img = cv2.resize(img,(widthImg,heightImg)) #Resize Image
#     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converting the gray scale
#     imgBlur = cv2.GaussianBlur(imgGray,(5,5),1) # ADD Gausian Blur
#     thres = utils.valTrackbars() # Get Trackbars values for Threshold

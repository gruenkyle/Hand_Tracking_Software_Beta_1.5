import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math

#########################

wCam, hCam = 640, 480      #Sets Webcam Width and Height

#########################

cap = cv2.VideoCapture(1)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector()

while True:

    #### Creates Webcam ####
    success, img = cap.read()

    #### Find Hand ####
    img = detector.findHands(img, True)
    lmList = detector.findPosition(img, draw=False)

    #### Print Thumb and Index Values ####

    if len(lmList) != 0:
        ##print (lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2


        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)  #Thumb Dot
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)  #Index Dot
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)     #Line Drawing

        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)   #Middle of Line Dot

        length = math.hypot(x2 - x1, y2 - y1)
        print (length)

        if length < 50:
            cv2.circle(img, (cx, cy), 5, (255, 255, 255), cv2.FILLED)  # Middle of Line Dot


    #### FPS Creator ####
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    #### Display Webcam ####
    cv2.imshow("Img", img)
    cv2.waitKey(1)
import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

cap = cv2.VideoCapture(1)

previousTime = 0
currentTime = 0

detector = htm.handDetector()

while True:
    # Creates Image that is being read
    success, img = cap.read()

    img = detector.findHands(img)

    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        print(lmList[4])

    #Tracks frames per second with real time
    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime

    #Prints FPS in corner of screen
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)

    #Webcam image being displayed on screen window
    cv2.imshow("Image", img)
    cv2.waitKey(1)

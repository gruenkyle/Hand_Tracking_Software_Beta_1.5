import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0

while True:
    #Creates Image that is being read
    success, img = cap.read()

    #Creates RGB Image
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #Draws lines on hand for tracking
    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(handLandmarks.landmark):
                #print(id, lm)
                h, w, c= img.shape
                px, py = int(lm.x*w), int(lm.y*h)

                print(id, px, py)


            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)


    #print(results.multi_hand_landmarks)

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



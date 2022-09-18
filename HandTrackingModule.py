import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.85, trackCon=0.85):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)

        #Uses self.mpHands instead since we are referencing the object on line 12

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        # Creates RGB Image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # Draws lines on hand for tracking
        if self.results.multi_hand_landmarks:
            for handLandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks,
                                                self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo= 0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:
            currHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(currHand.landmark):
                h, w, c = img.shape
                px, py = int(lm.x * w), int(lm.y * h)

                lmList.append([id, px, py])

                if draw:
                    if id == 4:
                        cv2.circle(img, (px, py), 10, (255, 0, 255), cv2.FILLED)
                    if id == 8:
                        cv2.circle(img, (px, py), 10, (255, 0, 255), cv2.FILLED)
                    if id == 12:
                        cv2.circle(img, (px, py), 10, (255, 0, 255), cv2.FILLED)
                    if id == 16:
                        cv2.circle(img, (px, py), 10, (255, 0, 255), cv2.FILLED)
                    if id == 20:
                        cv2.circle(img, (px, py), 10, (255, 0, 255), cv2.FILLED)

                #print(id, px, py)

        return lmList

    def openFingers(self, img, handNo=0):

        tipIds = [4, 8, 12, 16, 20]

        thu, ind, mid, rin, pin = False, False, False, False, False
        fingerList = [thu, ind, mid, rin, pin]

        lmList = self.findPosition(draw=False)

        for id in range(0, 5):
            index = 0
            fingerList[index] = lmList[[tipIds[id]][2]] < lmList[[tipIds[id] - 2]][2]

        print (fingerList)



def main():
    cap = cv2.VideoCapture(0)

    previousTime = 0
    currentTime = 0

    detector = handDetector()

    while True:
        # Creates Image that is being read
        success, img = cap.read()

        img = detector.findHands(img)

        lmList = detector.findPosition(img)

        detector.openFingers()

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



if __name__ == "__main__":
    main()


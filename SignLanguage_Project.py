import cv2
import time
import os
import HandTrackingModule as htm


def main():
    #########################
    wCam, hCam = 640, 480 # Sets Webcam Width and Height

    #### Set Up Webcam ####
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    #### Image Storage ####
    aslFolderPath = "Sign_Language_Images"
    imagesOfASL = os.listdir(aslFolderPath)
    print(imagesOfASL)

    overlayList = []
    for imPath in imagesOfASL:
        image = cv2.imread(f'{aslFolderPath}/{imPath}')  ## Gives Pathway Import for Searches
        overlayList.append(image)

    print(len(overlayList))

    #### Hand Detector ####
    detector = htm.handDetector()

    while True:

        #### Creates Webcam ####
        success, img = cap.read()

        #### Find Hand ####
        img = detector.findHands(img, True)

        lmList = detector.findPosition(img, draw=False)

        #### Find which fingers are closed ####
        if len(lmList) != 0:
            tipIds = [4, 8, 12, 16, 20]

            fingerList = []

            #The Thumb Finger
            if lmList[4][1] > lmList[3][1]:
                fingerList.append(1)
            else:
                fingerList.append(0)

            #The Other Fingers
            for id in range(1,5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                    fingerList.append(1)
                else:
                    fingerList.append(0)

            #### Interpret Which Letter ####

            # A, O
            if fingerList == [1, 0, 0, 0, 0]:
                h, w, c = overlayList[0].shape
                img[0:h, 0:w] = overlayList[0]

            elif fingerList == [0, 1, 1, 1, 1]:
                h, w, c = overlayList[1].shape
                img[0:h, 0:w] = overlayList[1]

            elif fingerList == [0, 1, 0, 0, 0]:
                h, w, c = overlayList[3].shape
                img[0:h, 0:w] = overlayList[3]

            elif fingerList == [0, 0, 0, 0, 0]:
                h, w, c = overlayList[4].shape
                img[0:h, 0:w] = overlayList[4]

            elif fingerList == [1,0,1,1,1]:
                h, w, c = overlayList[5].shape
                img[0:h, 0:w] = overlayList[5]


            print (fingerList)

        ######################################

        #### Display Webcam ####
        cv2.imshow("Img", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()

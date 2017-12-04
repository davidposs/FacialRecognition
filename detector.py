"""
Created on Thu Nov 16 16:27:29 2017
@author: David Poss

 Face Detection Program: reads faces from a web cam in the following steps:
    1. Get user's name to create a profile
    2. Take 11 pictures, one for each (optional) expression
    3. Saves to a data folder with other faces from the yale dataset
    4. Automatically resizes all images to 180x180
    It is recommended to only have 1 face in view at a time.
"""

import re
import os
import cv2

OUT_TYPE = ".jpg"
TEST_PATH = "../../Data/Test/"
TRAIN_PATH = "../../Data/Train/"


def main():
    """ This function can be used to train the face recognizer by gathering data for a user.
        Press 'q' to quit,
        Press 's' to save a face """

    name = input("Enter your name: ")
    cascades = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)
    known_faces = os.listdir(TRAIN_PATH)
    labels = known_faces

    labels = [re.findall(r'[^/]*$', i)[0].split(".")[0] for i in labels]

    while name in labels:
        print("That name is taken: choose another")
        name = input("Enter your name: ")

    for i in range(1, 21):
        print("taking picture {}".format(i))
        while True:
            _, frame = video_capture.read()
            grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascades.detectMultiScale(grayed, scaleFactor=1.1, minNeighbors=5)
            # Get x, y coordinates and width and height of face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.imshow('Webcam', frame)

            key_press = cv2.waitKey(1)
            # Press 's' to save detected face
            if key_press & 0xFF == ord('q'):
                break
            elif key_press & 0xFF == ord('s'):
                face_to_save = frame[y:y + h, x:x + w]
                cropped_face = cv2.resize(face_to_save, (180, 180), interpolation=cv2.INTER_AREA)
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                cv2.imshow("Save this?", cropped_face)
                # Separate into training and testing folders using 80-20 split
                if i < 5:
                    print("Save to {}".format(TEST_PATH + name + "." + str(i) + OUT_TYPE))
                    #cv2.imwrite(TEST_PATH + name + "." + str(i) + OUT_TYPE, cropped_face)
                else:
                    print("Save to {}".format(TRAIN_PATH + name + "." + str(i) + OUT_TYPE))
                    #cv2.imwrite(TRAIN_PATH + name + "." + str(i) + OUT_TYPE, cropped_face)
                break

    # Stop recording and close windows
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
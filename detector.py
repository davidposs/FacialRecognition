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
TRAIN_PATH = "../../Data/SmallerTrain/"
TEST_PATH = "../../Data/SmallerTest/"
PREDICT_PATH = "../../Data/"


def main():
    """ This function can be used to train the face recognizer by gathering data for a user.
        Press 'q' to quit,
        Press 's' to save a face """

    name = input("Enter your name: ")
    predict = input("Save an image for prediction? (y/n): ")

    cascades = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)
    labels = os.listdir(TRAIN_PATH)
    labels = [re.findall(r'[^/]*$', i)[0].split(".")[0] for i in labels]

    #while name in labels:
    #    print("That name is taken: choose another")
    #    name = input("Enter your name: ")


    for i in range(20, 40):
        print("taking picture {}".format(i))
        while True:
            _, frame = video_capture.read()
            # grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascades.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
            # Get x, y coordinates and width and height of face
            x_pos, y_pos, width, height = None, None, None, None
            for (x_pos, y_pos, width, height) in faces:
                cv2.rectangle(frame, (x_pos, y_pos), (x_pos + width, y_pos + height),
                              (0, 255, 255), 2)  # RGB values and thickness of detection box
            cv2.imshow('Webcam', frame)

            key_press = cv2.waitKey(1)
            # Press 's' to save detected face
            if key_press & 0xFF == ord('q'):
                break
            elif key_press & 0xFF == ord('s'):
                if x_pos is None or y_pos is None or width is None or height is None:
                    print("No face detected")
                    break
                face_to_save = frame[y_pos:y_pos + height, x_pos:x_pos + width]
                # Crop the face to 180 x 180
                face_to_save = cv2.resize(face_to_save, (180, 180), interpolation=cv2.INTER_AREA)
                # Convert to gray scale
                face_to_save = cv2.cvtColor(face_to_save, cv2.COLOR_BGR2GRAY)
                cv2.imshow("Save this?", face_to_save)
                # Separate into training and testing folders using 80-20 split
                if predict == "y":
                    print("Saving to {}".format(PREDICT_PATH + name + "." + str(454) + OUT_TYPE))
                    cv2.imwrite(PREDICT_PATH + name + "." + str(i) + OUT_TYPE, face_to_save)
                    video_capture.release()
                    cv2.destroyAllWindows()
                    return
                if i < 26:
                    print("Save to {}".format(TEST_PATH + name + "." + str(i) + OUT_TYPE))
                    cv2.imwrite(TEST_PATH + name + "." + str(i) + OUT_TYPE, face_to_save)
                else:
                    print("Save to {}".format(TRAIN_PATH + name + "." + str(i) + OUT_TYPE))
                    cv2.imwrite(TRAIN_PATH + name + "." + str(i) + OUT_TYPE, face_to_save)
                break

    # Stop recording and close windows
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

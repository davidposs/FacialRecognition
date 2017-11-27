# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:27:29 2017

@author: David
"""

""" Face Detection Program: reads faces from a web cam in the following steps:
    1. Get user's name to create a profile
    2. Take 11 pictures, one for each (optional) expression
    3. Saves to a data folder with other faces from the yale dataset
    4. Automatically resizes all images abitrarily to 161 x 161
    It is recommended to only have 1 face in view at a time. """

import os
import cv2

expressions = ["noglasses", "glasses", "centerlight", "wink", "happy" "sleepy",
               "leftlight", "surprised", "rightlight", "sad", "normal"]

out_type = ".jpg"
data_path = "../../Data/Data/"
out_path = "../../Data/NewFaces/"


def main():
    """ This function can be used to train the face recognizer by gathering data for a user.
        Press 'q' to quit,
        Press 's' to save a face """

    name = input("Enter your name: ")
    cascades = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)
    people_whose_face_i_already_know = os.listdir(data_path)
    # ids = sorted(list(set([int(re.findall(r'\d+', person)[0]) for person in people_whose_face_i_already_know])))
    labels = people_whose_face_i_already_know

    labels = [i.split(".")[0] for i in labels]
    while name in labels:
        print("That name is taken: choose another")
        name = input("Enter your name: ")

    # print("Hi {}, your unique ID is {}. Let's take some pictures. ".format(name, new_id))

    count = 0
    ###for i in range(20):
    for path in os.listdir(data_path):
        ###print("Taking picture number {}".format(i))
        ### print("Showing {}".format(data_path + path))
        while True:
            ###_, frame = video_capture.read()
            frame = cv2.imread(data_path + path)
            grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascades.detectMultiScale(grayed, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(20, 20))#, maxSize=(161, 161))
            # Get x, y coordinates and width and height of face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

            cv2.imshow('Webcam', frame)

            key_press = cv2.waitKey(1)
            # Press 's' to save detected face
            if key_press & 0xFF == ord('q'):
                break
            elif key_press & 0xFF == ord('s'):
                ###capture = name + "." + str(count) + out_type
                face_to_save = frame[y:y + h, x:x + w]
                ###print("Saving {}".format(out_path + capture))
                cropped_face = cv2.resize(face_to_save, (180, 180), interpolation=cv2.INTER_AREA)
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                cv2.imshow("Save this?", cropped_face)
                ###cv2.imwrite(out_path + capture,cropped_face)
                print("Saving {}".format(data_path + path))
                cv2.imwrite(out_path + path, cropped_face)
                count += 1
                break


    # Stop recording and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
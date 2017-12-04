"""
Created on Thu Nov 16 16:27:29 2017
@author: David Poss

Face Detection Program for use on existing images. Finds faces in them, crops them out and saves
 them to the specified output directories.

Tweak as needed to accommodate each dataset's peculiarities (such as labels or extensions).
"""

import os
import cv2

OUT_TYPE = ".jpg"
DATA_PATH = "../../Data/Test/"
OUT_PATH = "../../Data/Test/"


def main():
    """ This function can be used to train the face recognizer by gathering data for a user.
        Press 'q' to quit,
        Press 's' to save a face """

    cascades = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)
    for path in os.listdir(DATA_PATH):
        while True:
            frame = cv2.imread(DATA_PATH + path)
            grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascades.detectMultiScale(grayed, scaleFactor=1.1, minNeighbors=5,
                                              minSize=(20, 20))
            # Get x, y coordinates and width and height of face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.imshow('Image', frame)

            key_press = cv2.waitKey(1)
            # Press 's' to save detected face
            if key_press & 0xFF == ord('q'):
                break
            elif key_press & 0xFF == ord('s'):
                face_to_save = frame[y:y + h, x:x + w]
                cropped_face = cv2.resize(face_to_save, (180, 180), interpolation=cv2.INTER_AREA)
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                cv2.imshow("Save this?", cropped_face)
                print("Saving {}".format(DATA_PATH + path))
                cv2.imwrite(OUT_PATH + path, cropped_face)
                break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
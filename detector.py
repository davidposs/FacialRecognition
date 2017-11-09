""" Face Detection Program: reads faces from a web cam in the following steps:
    1. Get user's name to create a profile
    2. Take 11 pictures, one for each (optional) expression
    3. Saves to a data folder with other faces from the yale dataset
    It is recommended to only have 1 face in view at a time. """

import re
import os
import cv2

expressions = ["normal", "sad", "happy", "angry", "spooky", "confused",
               "surprised", "serious", "jealous", "distant", "laughing"]
out_type = ".jpg"
data_path = "../../Data/Faces/"

def main():
    """ This function can be used to train the face recognizer by gathering data for a user.
        Press 'q' to quit,
        Press 's' to save a face """

    name = input("Enter your name: ")
    cascades = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    video_capture = cv2.VideoCapture(0)
    people_whose_face_i_already_know = os.listdir(data_path)

    # Get a unique integer ID for a new user with name given above
    new_id = 1
    for person in people_whose_face_i_already_know:
        if new_id == int(re.findall(r'\d+', person)[0]):
            new_id += 1
    print ("Hi {}, your unique ID is {}. Let's take some pictures. ".format(name, new_id))
    for expression in expressions:
        print ("Show me {}".format(expression))
        while True:
            # Read from web cam in real time
            _, frame = video_capture.read()
            # Convert to gray scale to make recognition faster
            grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect a face in gray_frame
            faces = cascades.detectMultiScale(grayed, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            # Get x, y coordinates and width and height of face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.imshow('Webcam', frame)
            # Press 'q' to quit
            keypress = cv2.waitKey(1) & 0xFF
            # Press 's' to save detected face
            if keypress == ord('s'):
                #while os.path.exists(os.path.relpath("../../Data/Faces/" + capture + expression + out_type)):
                capture = name + str(new_id) + expression + out_type
                face_to_save = frame[y:y + h, x:x + w]
                print("Saving %s" % capture)
                #cv2.imwrite("../../Data/Faces/" + capture, face_to_save)
                break
            elif keypress == ord('q'):
                print ("Quitting...")
                return

    # Stop recording and close windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

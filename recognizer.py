""" Facial recognition program """

import os
import cv2
import sys
import re
import numpy as np

def find_face(image):
    # Find a face within an image and returned the gray-scaled window it is in
    grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascades = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # Grab the faces
    faces = cascades.detectMultiScale(grayed, scaleFactor=1.1, minNeighbors = 5)
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    return grayed[y:y+w, x:x+h], faces[0]


cascades = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8)

path_name = "../../Data/Faces/subject01centerlight.jpg"
image_paths = [os.path.join("../../Data/Faces/", f) for f in os.listdir("../../Data/Faces") if not f.endswith("centerlight.jpg")]
#print (path_name.find("Faces"))
#for im in image_paths:
#    label = int(re.findall(r'\d+', im)[0])
#    print (label)
    #name = im.split("/")[-1]  #+ str(label)
    #name = name[0:name.find(str(label)) + len(str(label))]
    #print (name + "              " + im + "\t\t\t " + str(label))
#    print (label)



#sys.exit()
#print (image_paths)

def preprocess(path, omit):
    """ Preprocess our data """
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith(omit)]
    faces = []
    labels = []
    # Go through each subject in the data folder
    for image_path in image_paths:
        # Get an image and apply a grayscale to it
        image = cv2.imread(image_path)
        #image = np.array(image, 'uint8')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # remove file extension to get label
        ##### label = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        image = np.array(image, 'uint8')
        label = int(re.findall(r'\d+', image_path)[0])
        #name = image_path.split("/")[-1]
        #name = name[0:name.find(str(label)) + len(str(label))]
        detected_faces = cascades.detectMultiScale(image)
        # Create a label for each image
        for (x, y, w, h) in detected_faces:
            faces.append(image[y:y+h, x:x+w])
            labels.append(label)

    return faces, labels

path = "../../Data/Faces/"
omit = "centerlight.jpg"
faces, labels = preprocess(path, omit)
print ("total faces: %s\ntotal labels %s" % (len(faces), len(labels)))
#sys.exit()
# Create Local Binary Pattern Histogram for faces
model.train(faces, np.array(labels))

def predict(image):
    img = image.copy()
    face, rect = find_face(img)
    label, confidence = model.predict(face)
    #label_text = persons[label]
    return label, confidence


print ("Prediction time!")
image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(omit)]
for image_path in image_paths:
    predict_image = cv2.imread(image_path)
    predict_image = cv2.cvtColor(predict_image, cv2.COLOR_BGR2GRAY)
    predict_image = np.array(predict_image, 'uint8')
    faces = cascades.detectMultiScale(predict_image)
    for (x, y, w, h) in faces:
        cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
        cv2.waitKey(1000)
        predict_label, confidence = model.predict(predict_image[y:y+h, x:x+w])
        #actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        actual = int(re.findall(r'\d+', image_path)[0])
        if predict_label == actual:
            print ("Correct: {}, {}".format(predict_label, confidence))
        else:
            print ("Incorrect :(: {}, {}".format(predict_label, actual))

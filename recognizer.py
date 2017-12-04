"""
Created on Thu Nov 16 16:27:29 2017
@author: David Poss

Facial Recognition Program using University of Essex Dataset
"""

from collections import OrderedDict
from datetime import datetime
import time
import tempfile
import os
import re
import cv2

import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 50
# All images are sized at 320x240
IMAGE_SIZE = (320, 240)
TRAIN_PATH = "../../Data/EyeTrain/"
TEST_PATH = "../../Data/EyeTest/"


def get_paths_and_labels(path):
    """ image_paths  :  list of relative image paths
        labels       :  mix of letters and numeric characters """
    image_paths = [path + image for image in os.listdir(path)]
    # labels = [i.split(".")[-3] for i in image_paths]
    # labels = [i.split("/")[-1] for i in labels]
    labels = [re.findall(r'[^/]*$', img_path)[0].split(".")[0] for img_path in image_paths]
    labels = [label[0:-2] for label in labels]
    return image_paths, labels


def weight_variable(shape):
    """ Create a weight variable for a given shape """
    initial = tf.truncated_normal(shape, stddev=1.0)
    return tf.Variable(initial)


def bias_variable(shape):
    """ Creates a bias variable for a given shape """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x_image, weights):
    """ Single step convolution, allows output to be same size as input """
    return tf.nn.conv2d(x_image, weights, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(matrix):
    """ Gets the max of each 2x2 grid and halves the dimensions of the image """
    return tf.nn.max_pool(matrix, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def gray(image):
    """ Converts image to grayscale """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize(image):
    """ Resizes an image to a size specified as a tuple """
    return cv2.resize(image, (IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def encode_labels(train_labels, test_labels):
    """ Assigns a numeric value to each label since some are subject's names """
    found_labels = []
    index = 0
    mapping = OrderedDict()
    for i in train_labels:
        if i in found_labels:
            continue
        mapping[i] = index
        index += 1
        found_labels.append(i)
    return [mapping[i] for i in train_labels], [mapping[i] for i in test_labels], mapping


def predict(pred_image, y_conv, x_images, keep, sess):
    """ Provided an input path, """
    prediction = tf.argmax(y_conv, 1)
    flat = gray(resize(cv2.imread(pred_image))).flatten()
    pred = prediction.eval(feed_dict={x_images: [flat], keep: 1.0}, session=sess)
    return pred


def main():
    """ Main CNN code """
    train_image_paths, train_labels = get_paths_and_labels(TRAIN_PATH)
    train_images = [gray(cv2.imread(image)).flatten() for image in train_image_paths]
    num_train_images = len(train_image_paths)
    print("Num train images {}".format(num_train_images))

    test_image_paths, test_labels = get_paths_and_labels(TEST_PATH)
    num_test_images = len(test_image_paths)
    print("Num test images {}".format(num_test_images))

    test_images = [gray(cv2.imread(image)).flatten() for image in test_image_paths]

    num_classes = len(set(train_labels))

    # Placeholders
    x_images = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE[0] * IMAGE_SIZE[1]])
    y_labels = tf.placeholder(tf.float32, shape=[None, num_classes])

    with tf.name_scope("Reshape"):
        x_image = tf.reshape(x_images, [-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1])

    # One-hot labels
    train_labels, test_labels, mapping = encode_labels(train_labels, test_labels)
    train_labels = tf.one_hot(indices=tf.cast(train_labels, tf.int32), depth=num_classes)
    test_labels = tf.one_hot(indices=tf.cast(test_labels, tf.int32), depth=num_classes)

    with tf.name_scope("Convolution1"):
        weight_conv1 = weight_variable([5, 5, 1, 64])
        bias_conv1 = bias_variable([64])
        h_conv1 = tf.nn.relu(conv2d(x_image, weight_conv1) + bias_conv1)

    with tf.name_scope("Pooling1"): # Pools 180x180 -> 90x90
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope("Convolution2"):
        weight_conv2 = weight_variable([5, 5, 64, 128])
        bias_conv2 = bias_variable([128])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, weight_conv2) + bias_conv2)

    with tf.name_scope("Pooling2"): # Pools 90x90 -> 45x45
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope("FC1"):
        weight_fc1 = weight_variable([80*60*128, 1024])
        bias_fc1 = bias_variable([1024])
        h_pool2_flattened = tf.reshape(h_pool2, [-1, 80*60*128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flattened, weight_fc1) + bias_fc1)

    with tf.name_scope("Dropout"):
        keep = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep)

    with tf.name_scope("FC2"):
        weight_fc2 = weight_variable([1024, num_classes])
        bias_fc2 = bias_variable([num_classes])
        y_conv = tf.matmul(h_fc1_drop, weight_fc2) + bias_fc2

    with tf.name_scope("CrossEntropy"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_labels, logits=y_conv))

    with tf.name_scope("Optimization"):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope("Accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    graph_loc = tempfile.mkdtemp()
    print("Saving graph to {}".format(graph_loc))
    writer = tf.summary.FileWriter(graph_loc)
    writer.add_graph(tf.get_default_graph())

    print("Training...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_labels = train_labels.eval()
        test_labels = test_labels.eval()
        for i in range(EPOCHS):
            start = time.time()
            print("Epoch {} started at {}".format(i, datetime.now().time()))
            for batch_num in range(num_train_images//BATCH_SIZE):
                batch_images = train_images[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
                batch_labels = train_labels[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
                if i % 2 == 1:
                    train_accuracy = accuracy.eval(
                        feed_dict={x_images:batch_images, y_labels: batch_labels, keep: 1.0})
                    print("Training accuracy : {}".format(train_accuracy))
                train_step.run(
                    feed_dict={x_images: batch_images, y_labels: batch_labels, keep: 0.5})

            end = time.time()
            print("Finished at {}, time elapsed {}\n\n".format(datetime.now().time(), end - start))

        # Run test set
        test_accuracy = accuracy.eval(
            feed_dict={x_images:test_images, y_labels:test_labels, keep:1.0})
        print("Test Accuracy {}".format(test_accuracy))

        # Predict
        prediction = predict("../../Data/aeval3.jpg", y_conv, x_images, keep, sess)
        print("Guessed {}, Correct {}".format(prediction, mapping['aeval']))


if __name__ == "__main__":
    main()

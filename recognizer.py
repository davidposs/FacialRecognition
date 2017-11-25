# -*- coding: utf-8 -*-
""" Facial Recognition Program using University of Essex Dataset """

import os
import cv2
import time
import tensorflow as tf
from datetime import datetime

EPOCHS = 10
BATCH_SIZE = 8
IMAGE_SIZE = (180, 180)
TRAIN_PATH = "../../Data/SmallTrain/"
TEST_PATH = "../../Data/SmallTest/"


def get_paths_and_labels(path):
    """ image_paths  :  list of relative image paths
        labels       :  mix of letters and numeric characters """
    image_paths = [path + image for image in os.listdir(path)]
    labels = [i.split(".")[-3] for i in image_paths]
    labels = [i.split("/")[-1] for i in labels]
    return image_paths, labels


def weight_variable(shape):
    """ Create a weight variable for a given shape """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """ Creates a bias variable for a given shape """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, weights):
    """ Single step convolution, allows output to be same size as input """
    return tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')


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
    mapping = {}
    for i in train_labels:
        if i in found_labels:
            continue
        mapping[i] = index
        index += 1
        found_labels.append(i)
    return [mapping[i] for i in train_labels], [mapping[i] for i in test_labels], mapping


def main():
    """ Main CNN code """
    train_image_paths, train_labels = get_paths_and_labels(TRAIN_PATH)
    train_images = [cv2.imread(image) for image in train_image_paths]
    num_train_images = len(train_image_paths)

    test_image_paths, test_labels = get_paths_and_labels(TEST_PATH)
    test_images = [cv2.imread(image) for image in test_image_paths]

    num_classes = len(set(train_labels))

    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE[0] * IMAGE_SIZE[1]])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])
    x_image = tf.reshape(x, [-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1])

    # One-hot labels
    ztrain_labels, ztest_labels, mapping = encode_labels(train_labels, test_labels)

    train_labels = tf.one_hot(indices=tf.cast(train_labels, tf.int32), depth=num_classes)
    test_labels = tf.one_hot(indices=tf.cast(test_labels, tf.int32), depth=num_classes)

    # Convolution 1
    weight_conv1 = weight_variable([5, 5, 1, 64])
    bias_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x_image, weight_conv1) + bias_conv1)

    # Pooling 1 : 180x180 -> 90x90
    h_pool1 = max_pool_2x2(h_conv1)

    # Convolution 2
    weight_conv2 = weight_variable([5, 5, 64, 32])
    bias_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weight_conv2) + bias_conv2)

    # Pooling 2 :  90x90 -> 45x45
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1
    w_fc1 = weight_variable([45*45*32, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flattened = tf.reshape(h_pool2, [-1, 45*45*32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flattened, w_fc1) + b_fc1)
    keep = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep)

    # Fully Connected Layer 2
    weight_fc2 = weight_variable([1024, num_classes])
    bias_fc2 = bias_variable([num_classes])
    y_conv = tf.matmul(h_fc1_drop, weight_fc2) + bias_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Training...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_labels = train_labels.eval()
        for i in range(EPOCHS):
            start = time.time()
            print ("Epoch {} started at {}".format(i, datetime.now().time()))
            for batch_num in range(num_train_images//BATCH_SIZE):
                # Read images, convert to gray-scale, resize to 180x180 and flatten them
                curr_batch = train_images[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
                curr_batch = [gray(image).flatten() for image in curr_batch]
                # Get labels for current batch
                curr_labels = train_labels[batch_num * BATCH_SIZE : (batch_num + 1) * BATCH_SIZE]
                sess.run([train_step], feed_dict={x:curr_batch, y_:curr_labels, keep:0.5})

            train_accuracy = accuracy.eval(feed_dict={x:curr_batch, y_: curr_labels, keep: 1.0})
            print("training accuracy : {}".format(train_accuracy))
            train_step.run(feed_dict={x: curr_batch, y_:curr_labels, keep: 0.5})
            end = time.time()
            print("Finished at {}, time elapsed {}\n\n".format(datetime.now().time(), end - start))

        print("Final train accuracy {}".format(accuracy.eval(
            feed_dict={x: curr_batch, y_: curr_labels, keep: 0.4})))
        # Run test set
        test_labels = test_labels.eval()
        test_images =[gray(image).flatten() for image in test_images]
        print("Test Accuracy {}".format(accuracy.eval(feed_dict={x:test_images, y_:test_labels, keep:1.0})))

        # Predict
        pred_image = "../../david16centerlight.jpg"
        prediction = tf.argmax(y_conv, 1)
        flat = gray(resize(cv2.imread(pred_image))).flatten()
        pred = prediction.eval(feed_dict={x:[flat], keep:1.0}, session=sess)
        print(pred)
        print("Correct: {}".format(mapping['davidposs']))


if __name__ == "__main__":
    main()

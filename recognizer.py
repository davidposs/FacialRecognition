# -*- coding: utf-8 -*-
""" Facial Recognition Program using Yale Face dataset """

import re
import os
import cv2
import tensorflow as tf

epochs = 30
batch_size = 10
IMAGE_SIZE = (160, 160)
data_path = "../../Data/Faces/"


def get_paths_and_labels(path):
    """ Returns:
            image_paths: list of relative image paths
            labels:      list of integers starting from 1 """
    image_paths = [path + image for image in os.listdir(path)]
    labels = [int(re.findall(r'\d+', image)[0]) for image in image_paths]
    return image_paths, labels

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def resize(image):
    return cv2.resize(image, (IMAGE_SIZE), interpolation=cv2.INTER_AREA)

def flatten(image):
    return image.flatten()

def gray_resize_and_flatten(image):
    return flatten(resize(gray(image)))

def main():
    all_images, all_labels = get_paths_and_labels(data_path)
    images = [cv2.imread(image) for image in all_images]
    num_images = len(all_images)

    unique_labels = []
    for label in all_labels:
        if label not in unique_labels:
            unique_labels.append(label)
    num_classes = len(unique_labels)

    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE[0] * IMAGE_SIZE[1]])
    y_ = tf.placeholder(tf.float32, shape=[None, num_classes])
    x_image = tf.reshape(x, [-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1])

    # One-hot labels
    onehot_labels = tf.one_hot(indices=tf.cast(all_labels, tf.int32), depth=num_classes)

    # Convolution 1
    weight_conv1 = weight_variable([5, 5, 1, 64])
    bias_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x_image, weight_conv1) + bias_conv1)

    # Pooling 1 : 160x160 -> 80x80
    h_pool1 = max_pool_2x2(h_conv1)

    # Convolution 2
    weight_conv2 = weight_variable([5, 5, 64, 32])
    bias_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weight_conv2) + bias_conv2)

    # Pooling 2 :  80x80 -> 40x40
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1
    w_fc1 = weight_variable([40*40*32, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flattened = tf.reshape(h_pool2, [-1, 40*40*32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flattened, w_fc1) + b_fc1)
    keep = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep)

    # Fully Connected Layer 2
    weight_fc2 = weight_variable([1024, len(unique_labels)])
    bias_fc2 = bias_variable([len(unique_labels)])
    y_conv = tf.matmul(h_fc1_drop, weight_fc2) + bias_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Training...")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        onehot_labels = onehot_labels.eval()
        for i in range(epochs):
            for batch_count in range(num_images//batch_size):
                # Read images, convert to grayscale and resize to 160*160 (by default they are 161*161)
                curr_batch = images[batch_count * batch_size : (batch_count + 1) * batch_size]
                curr_batch = [gray_resize_and_flatten(image) for image in curr_batch]
                # Get labels for current batch
                curr_labels = onehot_labels[batch_count * batch_size : (batch_count + 1) * batch_size]
                sess.run([train_step], feed_dict={x:curr_batch, y_:curr_labels, keep:0.1})
            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:curr_batch, y_: curr_labels, keep: 1.0})
                print("Epoch {} training accuracy : {}".format(i, train_accuracy))
            train_step.run(feed_dict={x: curr_batch, y_:curr_labels, keep: 0.1})

        print("Final train accuracy {}".format(accuracy.eval(feed_dict={x: curr_batch, y_: curr_labels, keep: 1.0})))

        # Predict
        pred_image = "../../david16centerlight.jpg"
        prediction = tf.argmax(y_conv,1)

        flat = gray_resize_and_flatten(cv2.imread(pred_image))
        pred = prediction.eval(feed_dict={x:[flat], keep:1.0}, session=sess)
        print(pred)


if __name__ == "__main__":
    main()

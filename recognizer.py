# -*- coding: utf-8 -*-
""" Facial Recognition Program """

import re
import os
import cv2
import tensorflow as tf

epochs = 30
batch_size = 16
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

def main():
    all_images, all_labels = get_paths_and_labels(data_path)
    images = [cv2.imread(image) for image in all_images]
    num_images = len(all_images)

    # image_tensors = [tf.image.encode_jpeg(image) for image in images]
    # image_dataset = tf.data.Dataset.from_tensors(image_tensors)

    unique_labels = []
    for l in all_labels:
        if l not in unique_labels:
            unique_labels.append(l)
    num_classes = len(unique_labels)

    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE[0] * IMAGE_SIZE[1]])
    y_ = tf.placeholder(tf.int32, [None, num_classes])

    # Convolution 1
    weight_conv1 = weight_variable([5, 5, 1, 32])
    bias_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, weight_conv1) + bias_conv1)

    # Pooling 1 : 160x160 -> 80x80
    h_pool1 = max_pool_2x2(h_conv1)

    # Convolution 2
    weight_conv2 = weight_variable([5, 5, 32, 64])
    bias_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, weight_conv2) + bias_conv2)

    # Pooling 2 :  80x80 -> 40x40
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1
    w_fc1 = weight_variable([40*40*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flattened = tf.reshape(h_pool2, [-1, 40*40*64])
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
        for i in range(epochs):
            for batch in range(num_images//batch_size):
                # curr_batch = image_tensors[batch*batch_size:(1+batch) * batch_size]

                # Read images, convert to grayscale and resize to 160*160 (by default they are 161*161)
                curr_batch = images[batch * batch_size : (batch + 1) * batch_size]
                gray_batch = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in curr_batch]
                resized_batch = [cv2.resize(i, IMAGE_SIZE, interpolation=cv2.INTER_AREA) for i in gray_batch]
                # Flatten images and feed into network
                flat_batch = [i.flatten() for i in resized_batch]
                curr_batch = flat_batch
                # Get labels
                curr_labels = all_labels[batch * batch_size : (batch + 1) * batch_size]
                sess.run([train_step], feed_dict={x:curr_batch, y_:curr_labels, keep:0.7})
            # curr_batch = next_batch(10, image_tensors, all_labels)
            if i % 10 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:curr_batch, y_: curr_labels, keep: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            train_step.run(feed_dict={x: curr_batch, y_:curr_labels, keep: 0.5})

        print("test accuracy %g" % accuracy.eval(feed_dict={x: curr_batch, y_: curr_labels, keep: 1.0}))

if __name__ == "__main__":
    main()
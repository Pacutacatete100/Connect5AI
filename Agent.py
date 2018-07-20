import tensorflow as tf
import numpy as np
import os
import random as rd
from PIL import  Image
from pyscreenshot import grab
import matplotlib.pyplot as plt
##import Discount


img = grab(bbox=(80, 70, 650, 650))
img2 = img.resize((320, 320), Image.ANTIALIAS)
img2.show()
class Agent:


    def __init__(self, color_channels, img_height, img_width, num_actions):
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[-1, img_width, img_height, color_channels])
        conv0 = tf.layers.conv2d(self.input_layer, filters=32, kernel_size=[5, 5], activation=tf.nn.relu, padding="SAME")
        conv1 = tf.layers.conv2d(conv0, filters=32, kernel_size=[5, 5], activation=tf.nn.relu, padding="SAME")
        conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=[5, 5], activation=tf.nn.relu, padding="SAME")
        dense0 = tf.layers.dense(conv2, 1024, activation=tf.nn.relu)
        out = tf.layers.dense(dense0, 1024, activation=tf.nn.relu)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
        self.train_operation = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

        self.outputs = tf.nn.softmax(out)
        self.choice = tf.argmax(self.outputs, axis=1)

        # Training Procedure
        self.rewards = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32)

        one_hot_actions = tf.one_hot(self.actions, num_actions)

        self.gradients_to_apply = []
        for index, variable in enumerate(tf.trainable_variables()):
            gradient_placeholder = tf.placeholder(tf.float32)
            self.gradients_to_apply.append(gradient_placeholder)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        self.update_gradients = optimizer.apply_gradients(zip(self.gradients_to_apply, tf.trainable_variables()))

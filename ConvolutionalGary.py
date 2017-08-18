from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import re, copy, math, time, random
import numpy as np
from PGNParser import *

import argparse
import sys

name = "CNNGary29"

FLAGS = None


def deepnn(x):
  fFilters = 64
  sFilters = 64
  dense = 8000
  dense2 = 6000
  
  x_image = tf.reshape(x, [-1, 8, 8, 11, 1])
  
  W_conv1 = weight_variable([3, 3, 3, 1, fFilters])
  b_conv1 = bias_variable([fFilters])
  h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)

  W_conv12 = weight_variable([3, 3, 3, fFilters, fFilters])
  b_conv12 = bias_variable([fFilters])
  h_conv12 = tf.nn.relu(conv3d(h_conv1, W_conv12) + b_conv12)

  W_conv13 = weight_variable([3, 3, 3, fFilters, fFilters])
  b_conv13 = bias_variable([fFilters])
  h_conv13 = tf.nn.relu(conv3d(h_conv12, W_conv13) + b_conv13)

  W_conv14 = weight_variable([2, 2, 2, fFilters, fFilters])
  b_conv14 = bias_variable([fFilters])
  h_conv14 = tf.nn.relu(conv3d(h_conv13, W_conv14) + b_conv14)
  
  h_pool1 = max_pool_2x2(h_conv14)

  W_conv2 = weight_variable([2, 2, 2, fFilters, sFilters])
  b_conv2 = bias_variable([sFilters])
  h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)

  W_conv22 = weight_variable([2, 2, 2, sFilters, sFilters])
  b_conv22 = bias_variable([sFilters])
  h_conv22 = tf.nn.relu(conv3d(h_conv2, W_conv22) + b_conv22)

  W_conv23 = weight_variable([1, 1, 1, sFilters, sFilters])
  b_conv23 = bias_variable([sFilters])
  h_conv23 = tf.nn.relu(conv3d(h_conv22, W_conv23) + b_conv23)
  
  W_conv24 = weight_variable([1, 1, 1, sFilters, sFilters])
  b_conv24 = bias_variable([sFilters])
  h_conv24 = tf.nn.relu(conv3d(h_conv23, W_conv24) + b_conv24)

  h_pool2 = max_pool_2x2(h_conv22)

  W_fc1 = weight_variable([2 * 2 * 11 * sFilters, dense])
  b_fc1 = bias_variable([dense])

  h_pool2_flat = tf.reshape(h_pool2, [-1, 2*2*11*sFilters])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc15 = weight_variable([dense, dense2])
  b_fc15 = bias_variable([dense2])
  out = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc15) + b_fc15)

  W_fc2 = weight_variable([dense2, 4096])
  b_fc2 = bias_variable([4096])

  y_conv = tf.matmul(out, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.avg_pool3d(x, ksize=[1, 2, 2, 1, 1],
                        strides=[1, 2, 2, 1, 1], padding='SAME')


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  x = tf.placeholder(tf.float32, [None, 704])

  y_ = tf.placeholder(tf.float32, [None, 4096])

  y_conv, keep_prob = deepnn(x)

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "/tmp/CNNGary29.ckpt")
    print("Model restored.")
    currentGame = ""; tCost = 0; s = 0; e = 0; sCost = 0; moves = 0; timer = time.time()
    
    limit = 1300000
    for i in ["CheckGamesOnline.txt","CheckGamesHQ.txt","HQData.txt"]:
      file = open("Match Data/"+i,"r",encoding="latin-1")
      print("File loaded")
      for line in file:
          if s == limit:
            break
          if line[0] not in ["[","\n"]:
              currentGame += line[:-1] + " "
          if line[0] == "[" and currentGame != "":
              if "{" not in currentGame and "}" not in currentGame and "..." not in currentGame:
                  board = Board(); error = False
                  try:
                    d, a, b, w, wcms, bcms = parseText(currentGame[:-1],board)

                  except:
                      error = True
                      e+=1
                      
                  if error == False:
                      s+=1
                      
                      inData,half = prepData(d,w,wcms,bcms)
                      outData = mutEx(a,b,w,half)

                      batch_x = np.reshape(np.asarray(inData),(len(inData),704))
                      batch_y = np.reshape(np.asarray(outData),(len(outData),4096))

                      _, c = sess.run([train_step, cross_entropy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.75})
                      tCost += c
                      sCost += c
                      moves += len(inData)
                  
                      
              currentGame = ""
              if s % 100 == 0:
                  print("Cost=", "{:.2f}".format(c)," Average cost: ","{:.2f}".format(tCost/100), "Game: ",s, "Moves: ",moves," Errors: ",e, "Time: ","{:.2f}".format(time.time()-timer))
                  tCost = 0; timer = time.time()
              if s % 1000 == 0:
                  print("\n")
              if s % 5000 == 0:
                  save_path = saver.save(sess, "/tmp/"+name+".ckpt")
                  print("Model saved in file: %s" % save_path)
                  print("Average cost (last 5000): ",sCost/5000)
                  print("Total positions evaluated: ",moves)
                  sCost = 0
      
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

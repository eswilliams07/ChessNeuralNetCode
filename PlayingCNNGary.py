from __future__ import print_function

import tensorflow as tf
import re, copy, math, time
import numpy as np
from PGNParser import *

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
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.avg_pool3d(x, ksize=[1, 2, 2, 1, 1],
                        strides=[1, 2, 2, 1, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

x = tf.placeholder(tf.float32, [None, 704])

y_ = tf.placeholder(tf.float32, [None, 4096])

y_conv, keep_prob = deepnn(x)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
colour = 0

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "/tmp/CNNGary29.ckpt")
    print("Model restored.")

    board = Board()
    board.printBoard()
    turn = 1
    
    while True:
        if turn%2 == 0:
            colour = 1
        else:
            colour = 0

        whiteCMap = createCM(board.board,0)
        blackCMap = createCM(board.board,1)
        inp = dim(board.board)
        for i in range(8):
            for j in range(8):
                inp[i][j].append(colour)
                inp[i][j].append(whiteCMap[i][j])
                inp[i][j].append(blackCMap[i][j])

        inp = (np.reshape(np.asarray(inp),(1,704)))
        
        move = y_conv.eval(feed_dict={x: inp, keep_prob: 1})

        n = reverseExF(tf.nn.softmax(move.flatten()).eval())

        iMoves = []
        while True:
            tBoard = copy.deepcopy(board)
            index, p = exEvalMove(n,tBoard,iMoves)

            pos = [chr(index[1]+97),8-index[0]]
            dest = [chr(index[3]+97),8-index[2]]

            print(pos)
            print(dest)
            print(p*100)

            print(tBoard.board[index[0]][index[1]].type)
            t = tBoard.board[index[0]][index[1]].type
            if t != 6:
                tBoard.board[index[0]][index[1]] = 0
                tBoard.board[index[2]][index[3]] = Piece(t,colour,index[2],index[3])
            else:
                if abs(index[3]-index[1]) > 1:
                    direction = (index[3]-index[1])/abs(index[3]-index[1])
                    if direction > 0:
                        if colour == 0:
                            rook = tBoard.board[7][7]
                            tBoard.board[7][7] = 0
                            tBoard.board[7][5] = rook
                        else:
                            rook = tBoard.board[0][7]
                            tBoard.board[0][7] = 0
                            tBoard.board[0][5] = rook

                    if direction < 0:
                        if colour == 0:
                            rook = tBoard.board[7][0]
                            tBoard.board[7][0] = 0
                            tBoard.board[7][3] = rook
                        else:
                            rook = tBoard.board[0][0]
                            tBoard.board[0][0] = 0
                            tBoard.board[0][3] = rook
                
                tBoard.board[index[0]][index[1]] = 0
                tBoard.board[index[2]][index[3]] = Piece(t,colour,index[2],index[3])
                

            print("\n")
            tBoard.printBoard()

            k = input()
            if k != "I":
                board = tBoard
                break
            else:
                iMoves.append(index)

        pMove = input("Please enter move: ")
        pMove = Move(pMove,1)
        board.applyMove(pMove)
        time.sleep(1)


# An undercomplete autoencoder on MNIST dataset
from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
# from tensorflow.examples.tutorials.mnist import input_data
import pickle
import particle
import sys
import os
import IPython
import random
import time

if len(sys.argv) <= 1:
    name = 'mc'
else:
    name = sys.argv[1]
# Echo the example name.
print("Running example", name)


def PlotData(data):
    _, ax = plt.subplots(1, 1)
    ax.set_title("data")
    PlotMicrostructure(ax, data)
    plt.show()


def PlotMicrostructure(ax, data):
    # shape is a 2d numpy matrix.
    ax.clear()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            nodes = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            scale = 1. / data.shape[0]
            nodes = [(n[0] * scale, n[1] * scale) for n in nodes]
            #density = 1 if data[i, j] > 0.5 else 0
            density = max(min(data[i, j], 1.0), 0.0)
            ax.add_patch(plt.Polygon(nodes, color=(density, density, density)))


def ReadMicrostructure():
    # Load and visualize the microstructure.
    t0 = time.time()
    N = 10000
    if os.path.isfile('data_40.pkl'):
        with open('data_40.pkl', 'rb') as f:
            data = pickle.load(f)
        data = data[:N, ...]
    else:
        with open('data_40.pkl', 'rb') as f:
            particles = pickle.load(f)
        random.shuffle(particles)
        particles = particles[:]
        t0 = time.time()
        shape = particles[0].configuration.shape
        data = np.zeros((N, shape[0] * 2, shape[1] * 2))
        for i in range(N):
            # Supersampling.
            for ii in range(2 * shape[0]):
                for jj in range(2 * shape[1]):
                    data[i][ii][jj] = particles[i].configuration[ii // 2][jj // 2]
        pickle.dump(data, open("data_40.pkl", "wb"))
    t1 = time.time()
    print('Preparing the data ...', t1 - t0)
    return data

def GenerateCube():
    t0 = time.time()
    N = 10000
    m = 40
    data = np.zeros((N, m, m))
    for i in range(N):
        loc = (np.random.rand(2, 1) * 0.6) * (m - 1)
        x = max(min(int(loc[0]), m - 1), 0)
        y = max(min(int(loc[1]), m - 1), 0)
        s = max(int((np.random.rand() * 0.5 + 0.25) * (m - max(x, y))), 1)
        data[i][x:x+s,y:y+s] = 1.0
    t1 = time.time()
    print('Preparing the data ...', t1 - t0)
    print('Average area:', data.sum() / N)
    return data

if name == "mc":
    data = ReadMicrostructure()
elif name == "cube":
    data = GenerateCube()
else:
    raise NotImplementedError

m = data.shape[1]
N = data.shape[0]
data = data.reshape((-1, m, m, 1))
N_train = int(N * 0.8)
N_test = N - N_train
training_data = data[:N_train, :]
test_data = data[N_train:, :]

batch_size = 500   # Number of samples in each batch
epoch_num = 100    # Number of epochs to train the network
lr = 0.001         # Learning rate


def autoencoder(inputs):
    if name == "mc":
        # Encoder.
        # 40 x 40 x 1 -> 20 x 20 x 32
        # 20 x 20 x 32 -> 10 x 10 x 16
        # 10 x 10 x 16 -> 5 x 5 x 8
        net = lays.conv2d(inputs, 32, [3, 3], stride=2, padding='same', activation_fn=tf.nn.tanh)
        net = lays.conv2d(net, 16, [3, 3], stride=2, padding='same', activation_fn=tf.nn.tanh)
        net = lays.conv2d(net, 8, [3, 3], stride=2, padding='same', activation_fn=tf.nn.tanh)

        # Decoder.
        net = lays.conv2d_transpose(net, 16, [3, 3], stride=2, padding='same', activation_fn=tf.nn.tanh)
        net = lays.conv2d_transpose(net, 32, [3, 3], stride=2, padding='same', activation_fn=tf.nn.tanh)
        net = lays.conv2d_transpose(net, 1, [3, 3], stride=2, padding='same', activation_fn=tf.nn.tanh)
    elif name == "cube":
        inputs = tf.reshape(inputs, [-1, 1600])
        net = tf.layers.dense(inputs, 400, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 100, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 10, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 3, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 10, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 100, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 400, activation=tf.nn.tanh)
        net = tf.layers.dense(net, 1600, activation=tf.nn.tanh)
        net = tf.reshape(net, [-1, 40, 40, 1])
    else:
        raise NotImplementedError
    # Rescale to [0, 1].
    net = net * 0.5 + 0.5
    return net


batch_per_ep = N_train // batch_size
ae_inputs = tf.placeholder(tf.float32, (None, m, m, 1))
ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network

# calculate the loss and optimize the network
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # calculate the mean square error loss
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

# initialize the network
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    losses = []
    diff_pixels = []
    for ep in range(epoch_num):  # epochs loop
        l = []
        for batch_n in range(batch_per_ep):  # batches loop
            batch_data = training_data[batch_n*batch_size:(batch_n+1)*batch_size, ...]
            _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_data})
            print('Epoch: {} - cost = {:.5f}'.format((ep + 1), c))
            l.append(c)
        test_output = sess.run(ae_outputs, feed_dict={ae_inputs: test_data})
        n = 0
        for i in range(test_output.shape[0]):
            a = test_data[i].flatten()
            b = test_output[i].flatten()
            b[b > 0.5] = 1.0
            b[b <= 0.5] = 0.0
            n += np.sum(np.abs(a - b))
        n /= (test_output.shape[0] * m * m)
        diff_pixels.append(n)
        losses.append(sum(l) / len(l))

    # Display the epoch-loss.
    _, ax = plt.subplots()
    ax.set_title('Training error')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.plot(list(range(len(losses))), losses, label='test loss')
    ax.plot(list(range(len(losses))), diff_pixels, label='different pixels')
    ax.legend()
    plt.show()

    # Test the autoencoder.
    test_output, test_loss = sess.run([ae_outputs, loss], feed_dict={ae_inputs: test_data})
    n = 0
    for i in range(test_output.shape[0]):
        a = test_data[i].flatten()
        b = test_output[i].flatten()
        b[b>0.5]=1.0
        b[b<=0.5]=0.0
        n += np.sum(np.abs(a - b))
    n /= (test_output.shape[0] * m * m)
    print('Test loss:', test_loss, 'different pixels:', n)

    # Display the test results.
    def Display(i):
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.set_title("Input microstructure")
        PlotMicrostructure(ax0, test_data[i].reshape(m, m))
        ax1.set_title("Reconstructed microstructure")
        PlotMicrostructure(ax1, test_output[i].reshape(m, m))
        ax2.set_title("Reconstructed after clamping")
        D = test_output[i].reshape(m, m)
        D[D>0.5] = 1.0
        D[D<=0.5] = 0.0
        PlotMicrostructure(ax2, D)
        plt.show()
    IPython.embed()

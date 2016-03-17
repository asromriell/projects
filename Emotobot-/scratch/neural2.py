# Students: D. Wen, A. Romriell, J. Pollard
# MSAN 630 ML2 Project

"""
This is the main file for our CNN implementation. We currently do not
have a working model.
Our plan is to implement a simple 3 convolutional/pooling layer followed by
a standard neural network that outputs a probability for each of the seven
emotion classes.
"""

import os
import re
import time
import random
import emotions
from glob import glob
import itertools
import argparse
import numpy as np
from numpy.random import uniform
from scipy.signal import convolve2d
import skimage
from skimage import io
from skimage.measure import block_reduce


def sigmoid(x):

    return 1.0 / (1 + np.exp(-x))


def parse_arguments():
    """
    -parses the command line arguments and returns a dictionary with
    the train file directory
    :return:
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", required=True)
    arguments = arg_parser.parse_args()

    # using the vars() function allows the return to be a dictionary
    # of the form {'path': directory where training images are located}
    return vars(arguments)



def shuffle_images(dir):
    """
    -returns a list of files that have been shuffled in place randomly
    """
    random.seed(1)
    image_filenames = glob('{}/*.png'.format(dir))
    image_filenames.sort()
    random.shuffle(image_filenames)
    return image_filenames



# def get_emotion(png_files):
#     """
#     get the emotion from the filename
#     :param path:
#     :return:
#     """
#     # onlyfiles = [join(path,f) for f in listdir(path) if isfile(join(path, f))]
#     # png_files = [f for f in onlyfiles if f.endswith(".png")]
#
#     for f in png_files:
#         dir, file = os.path.split(f)
#         e = emotions.Emotions()
#         print e.build_vector(file[0])


def get_emotion(k):

    em = emotions.Emotions()
    return em.build_vector(k[0])


def load_image(png_files):
    """
    -creates a dictionary of filename: image
    """
    img_dict = dict()

    for f in png_files:
        img_key = f.split('/')[-1].split('.')[0]
        img = io.imread(f, as_grey=True)
        img_dict[img_key] = img

    return img_dict


def convolve(img, ker):

    return convolve2d(img, ker, mode="same", boundary="symm")


def pool(img, pool_size):

    return block_reduce(img, (pool_size, pool_size), func=np.max)


def network(img, emo):

    err = 0
    tot = 0

    kernel = uniform(0, 0.001, (3, 3))

    c = convolve(img, kernel)

    p = pool(c, 8)

    x = p.flatten()

    w = uniform(0, 0.001, 36)

    z = sigmoid(np.dot(x, w))

    alphas = uniform(0, 0.001, 7)

    t = z * alphas

    # print t
    # print sigmoid(t)
    # print emo
    # print np.argmax(sigmoid(t))
    # print np.argmax(emo)
    #
    # if np.argmax(sigmoid(t)) == np.argmax(emo):
    #     print "not doing so bad..."
    #
    # else:
    #     print "your code sucks, dude"
    #     err += 1

    print "now let's try softmax..."

    s = np.sum(np.exp(t))
    out = np.array([np.exp(el)/s for el in t])

    print out
    print emo
    print np.argmax(out)
    print np.argmax(emo)

    if np.argmax(out) == np.argmax(emo):
        print "not doing so bad..."

    else:
        print "your code still sucks, dude"


"""
TODO:
2. write function implement the convolutional and neural layers
3. compute gradients for both the neurons and estimated kernels and implement gradient descent
   to update the weights
4. perform leave-out-10% training/validation
"""




if __name__ == "__main__":

    start = time.time()

    print "Testing..."
    args = parse_arguments()

    ims = load_image(shuffle_images(args['path']))
    print "There are {} images in the list...".format(len(ims.keys()))

    test_img = ims[ims.keys()[0]]
    print ims.keys()[0]

    network(test_img, get_emotion(ims.keys()[0]))






    print "\nThis took {0} seconds to complete...".format(time.time()-start)


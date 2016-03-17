# Students: David Wen, Alex Romriell, Jacob Pollard
# MSAN 630 ML 2 Emotobot Project

import time
from skimage import io
from skimage import transform
from scipy import ndimage
from skimage.util import crop
import numpy as np
import theano
import theano.tensor as T
import lasagne

IMDIR = "/Users/jtpollard/MSAN/msan630/Project/our_images/"
# KERNELS = "/Users/jtpollard/MSAN/msan630/Project/trained_kernels/large_3072_rescale_disgust_sad.npz"
KERNELS = "/Users/jtpollard/MSAN/msan630/Project/trained_kernels/four_cnn_fc_4096_rescale_happy_disgust.npz"
# KERNELS = "/Users/jtpollard/MSAN/msan630/Project/trained_kernels/large_4096_rescale_happy_sad.npz"
EMOS = {"A": 0, "D": 1, "F": 2, "H": 3, "U": 4, "S": 5, "N": 6}
EM_ABBREV = {"Angry": "A", "Disgust": "D", "Fear": "F", "Happy": "H", "Unhappy": "U", "Surprise": "S", "Neutral": "N"}


def rescale_image(image):

    rows, columns = np.shape(image)
    rescale = [[float(image[r, c]) / 255 for c in range(columns)] for r in range(rows)]

    return np.array(rescale)


def get_our_images(imdir):
    """
    -resize our images to 48X48 which is what the model is expecting
    :param imdir:
    :return:
    """
    jacob = imdir + 'HJACOB.png'
    alex = imdir + 'HALEX.png'
    david = imdir + 'HDAVID.png'
    yannet = imdir + 'HYANNET.png'
    james = imdir + 'HJAMES.png'
    paul = imdir + 'DPAUL.png'

    # load us in
    X_jacob_orig = io.imread(jacob, as_grey=True)
    X_alex_orig = io.imread(alex, as_grey=True)
    X_david_orig = io.imread(david, as_grey=True)
    X_yannet_orig = io.imread(yannet, as_grey=True)
    X_james_orig = io.imread(james, as_grey=True)
    X_paul_orig = io.imread(paul, as_grey=True)

    # resize to 48 X 48
    X_jacob_orig = transform.resize(X_jacob_orig, [48, 48])
    X_alex_orig = transform.resize(X_alex_orig, [48, 48])
    X_david_orig = transform.resize(X_david_orig, [48, 48])
    X_yannet_orig = transform.resize(X_yannet_orig, [48, 48])
    X_james_orig = transform.resize(X_james_orig, [48, 48])
    X_paul_orig = transform.resize(X_paul_orig, [48, 48])

    # smooth the image a tiny bit
    X_jacob_orig = ndimage.median_filter(X_jacob_orig, 1.5)
    X_alex_orig = ndimage.median_filter(X_alex_orig, 1.5)
    X_david_orig = ndimage.median_filter(X_david_orig, 1.5)
    X_yannet_orig = ndimage.median_filter(X_yannet_orig, 1.5)
    X_james_orig = ndimage.median_filter(X_james_orig, 1.5)
    X_paul_orig = ndimage.median_filter(X_paul_orig, 1.5)

    # crop the image down to 42 X 42
    X_jacob_orig = crop(X_jacob_orig, crop_width=3)
    X_alex_orig = crop(X_alex_orig, crop_width=3)
    X_david_orig = crop(X_david_orig, crop_width=3)
    X_yannet_orig = crop(X_yannet_orig, crop_width=3)
    X_james_orig = crop(X_james_orig, crop_width=3)
    X_paul_orig = crop(X_paul_orig, crop_width=3)

    # rescale the image to [0, 1] scale
    X_jacob = rescale_image(X_jacob_orig)
    X_alex = rescale_image(X_alex_orig)
    X_david = rescale_image(X_david_orig)
    X_yannet = rescale_image(X_yannet_orig)
    X_james = rescale_image(X_james_orig)
    X_paul = rescale_image(X_paul_orig)

    originals = np.empty(shape=(6, 1, 42, 42))
    our_test = np.empty(shape=(6, 1, 42, 42), dtype=np.float32)
    our_emos = np.empty(shape=6, dtype=np.int8)

    # original images resized
    originals[0, 0] = X_jacob_orig
    # io.imshow(originals[0,0])
    # io.show()
    originals[1, 0] = X_alex_orig
    originals[2, 0] = X_david_orig
    originals[3, 0] = X_yannet_orig
    originals[4, 0] = X_james_orig
    originals[5, 0] = X_paul_orig

    # inputs
    our_test[0, 0] = X_jacob
    our_test[1, 0] = X_alex
    our_test[2, 0] = X_david
    our_test[3, 0] = X_yannet
    our_test[4, 0] = X_james
    our_test[5, 0] = X_paul

    # input classifications
    our_emos[0, ] = 0
    our_emos[1, ] = 0
    our_emos[2, ] = 0
    our_emos[3, ] = 1
    our_emos[4, ] = 0
    our_emos[5, ] = 1

    return originals, our_test, our_emos


def prebuild_cnn1(input_var=None, params=None):

    # input layer is 42 X 42 image
    network = lasagne.layers.InputLayer(shape=(None, 1, 42, 42), input_var=input_var)

    # conv1 layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(7, 7),
                                         stride=1, pad=3,
                                         nonlinearity=lasagne.nonlinearities.rectify)
    # max pool layer 1
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2, pad=0)

    # conv2 layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(7, 7),
                                         stride=1, pad=3,
                                         nonlinearity=lasagne.nonlinearities.rectify)
    # max pool layer 2
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2, pad=(1, 0))

    # conv3 layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=128, filter_size=(7, 7),
                                         stride=1, pad=3,
                                         nonlinearity=lasagne.nonlinearities.rectify)
    # max pool layer 3
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2, pad=0)

    # conv4 layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=256, filter_size=(7, 7),
                                         stride=1, pad=3,
                                         nonlinearity=lasagne.nonlinearities.rectify)
    # max pool layer 4
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2, pad=0)

    # enter the fully connected hidden layer 4096 neurons
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=0.5),
                                        num_units=4096,
                                        nonlinearity=lasagne.nonlinearities.sigmoid)

    # again the num_units=7 refers to the 7 emotions we are classifying
    network = lasagne.layers.DenseLayer(network, num_units=2,
                                        nonlinearity=lasagne.nonlinearities.softmax)

    lasagne.layers.set_all_param_values(network, params)

    return network


def prebuild_cnn2(input_var=None, params=None):

    # input layer is 42 X 42 image
    network = lasagne.layers.InputLayer(shape=(None, 1, 42, 42), input_var=input_var)

    # conv1 layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(7, 7),
                                         stride=1, pad=3,
                                         nonlinearity=lasagne.nonlinearities.rectify)
    # max pool layer 1
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2, pad=0)

    # conv2 layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(7, 7),
                                         stride=1, pad=3,
                                         nonlinearity=lasagne.nonlinearities.rectify)
    # max pool layer 2
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2, pad=(1, 0))

    # conv3 layer
    network = lasagne.layers.Conv2DLayer(network, num_filters=64, filter_size=(7, 7),
                                         stride=1, pad=3,
                                         nonlinearity=lasagne.nonlinearities.rectify)
    # max pool layer 3
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2), stride=2, pad=0)

    # enter the fully connected hidden layer 4096 neurons
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=0.5),
                                        num_units=4096,
                                        nonlinearity=lasagne.nonlinearities.sigmoid)

    # again the num_units=7 refers to the 7 emotions we are classifying
    network = lasagne.layers.DenseLayer(network, num_units=2,
                                        nonlinearity=lasagne.nonlinearities.softmax)

    # set the pre-trained parameters from the last layer back
    lasagne.layers.set_all_param_values(network, params)

    return network



if __name__ == "__main__":

    start = time.time()

    # this is our pictures and a few of our professors'
    ours, our_emos = get_our_images(IMDIR)

    images = T.tensor4('inputs')
    emos = T.ivector('targets')

    # get our pre-trained parameters
    with np.load(KERNELS) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]

    for i in range(len(param_values)):
        print np.shape(param_values[i])





    # # create the network
    # emo_cnn = prebuild_cnn2(images, param_values)
    # pred = lasagne.layers.get_output(emo_cnn, deterministic=True)
    # th_func = theano.function([images, emos], [emos, pred])

    # predicting on our pictures
    # our_pred, th_pred = th_func(ours, our_emos)
    # print our_pred
    # print th_pred

    # loss = lasagne.objectives.categorical_crossentropy(pred, emos)
    # loss = loss.mean()
    # params = lasagne.layers.get_all_params(emo_cnn, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.004, momentum=0.9)
    # test_prediction = lasagne.layers.get_output(emo_cnn, deterministic=True)
    # test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, emos)
    # test_loss = test_loss.mean()
    # test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), emos), dtype=theano.config.floatX)
    #
    # train_func = theano.function([images, emos], loss, updates=updates)
    # # val_func = theano.function([images, emos], [test_loss, test_acc])
    # test_func = theano.function([images, emos], [test_loss, test_acc])


    print "This took {} seconds to run...".format(time.time() - start)


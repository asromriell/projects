# __author__ = 'alexromriell'

from sklearn import svm, preprocessing, decomposition
from skimage import io, transform
from sklearn.cross_validation import train_test_split
import numpy as np
import glob
import time
import csv
from sklearn.externals import joblib
import os
from scipy import ndimage


EMOS = {"A": 0, "D": 1, "F": 2, "H": 3, "U": 4, "S": 5, "N": 6}


def get_emo_int(filename):
    """
    -returns an integer between 0 and 6, inclusive, to identify one of the six emotions
    :param filename:
    :return:
    """
    pieces = filename.split('/')
    emo_key = pieces[-1][0]

    return EMOS[emo_key]


def create_train_test(filepath, emos=None):

    if emos is None:
        file_list = glob.glob(filepath + '/*.png')

    else:
        file_list = glob.glob(filepath + '/' + emos[0] + '*.png')
        file_list += glob.glob(filepath + '/' + emos[1] + '*.png')
        # file_list += glob.glob(filepath + '/' + emos[2] + '*.png')

    l = len(file_list)
    y = np.zeros(l)
    X = np.empty(shape=(l, 48, 48), dtype=np.int32)

    for i in range(l):

        tmp = io.imread(file_list[i], as_grey=True)  # read in image
        X[i] = tmp  #.flatten()           # don't flatten for PCA's
        y[i] = get_emo_int(file_list[i])

    dims = X.shape

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    return X_train, X_test, y_train, y_test, dims


def get_pcs(train, test, ncomps=15):
    pca_train = np.zeros(shape=(len(train), 48, 48), dtype=np.float64)
    pca_test = np.zeros(shape=(len(test), 48, 48), dtype=np.float64)

    for i in range(len(train)):
        # print train[i]
        # Standardize features by removing the mean and scaling to unit variance
        scaler_tr = preprocessing.StandardScaler().fit(train[i].astype('float64'))

        # Perform standardization by centering and scaling
        transformed_tr = scaler_tr.transform(train[i].astype('float64'))

        # create the PCA classifier object
        pcomp_tr = decomposition.PCA()

        # fit the PCA object with the transformed test/train data
        pcomp_tr.fit(train[i])

        # assign the PC's to the pca_train X_train data
        pca_train[i] = pcomp_tr.components_

    for i in range(len(test)):
        scaler_te = preprocessing.StandardScaler().fit(test[i].astype('float64'))
        transformed_te = scaler_te.transform(test[i].astype('float64'))
        pcomp_te = decomposition.PCA()
        pcomp_te.fit(test[i])
        pca_test[i] = pcomp_te.components_

    pca_train = pca_train[:, :ncomps:]  # take first ncomps
    pca_test = pca_test[:, :ncomps:]

    pca_train = pca_train.reshape(len(pca_train), ncomps*48)  # flatten into ncomps * 48
    pca_test = pca_test.reshape(len(pca_test), ncomps*48)

    return pca_train, pca_test


def train_svm(X, y, kernel='rbf', degree=3, gamma=0.001):
    clf = svm.SVC(kernel=kernel, degree=degree, gamma=gamma)
    clf.fit(X, y)
    return clf


if __name__ == '__main__':

    path = '/Users/alexromriell/Documents/DATA/AdvMLData/perturbed_images/all_images'
    # path = '/Users/alexromriell/Documents/DATA/AdvMLData/subset_1000'
    # path = '/Users/alexromriell/Documents/DATA/AdvMLData/subset_10000'
    # path = '/Users/alexromriell/Documents/DATA/AdvMLData/subset_20000'
    # path = '/Users/alexromriell/Documents/DATA/AdvMLData/all_images'
    log_file = '/Users/alexromriell/Dropbox/School/SpringModule1/AdvancedML/Project/svm_log.csv'

    t0 = time.time()
    emotions = ['H', 'N']

    X_train, X_test, y_train, y_test, dims = create_train_test(path, emos=emotions)

    # reassign train/test to PC's
    X_train, X_test = get_pcs(X_train, X_test, ncomps=15)

    kernel = 'rbf'
    degree = 3
    svm_clf = train_svm(X_train, y_train, kernel=kernel, degree=degree)

    train_accuracy = round(svm_clf.score(X_train, y_train), 3)
    test_accuracy = round(svm_clf.score(X_test, y_test), 3)
    runtime = round((time.time() - t0) / 60, 3)
    path = path.split('/')

    print dims[0:2]
    print "Train Accuracy:", train_accuracy
    print "Test Accuracy:", test_accuracy
    print "Total Time (min)", runtime

    with open(log_file, 'a') as f2:
                    a = csv.writer(f2, delimiter=',')
                    data = [[emotions, dims[0:2], kernel, degree, test_accuracy,
                             train_accuracy, runtime, path[-2] + '/' + path[-1], 'pca']]
                    a.writerows(data)

    # Pickle the model parameters...
    subdir = emotions[0] + '_' + emotions[1] + '_' + path[-1] + kernel + '_PC15all'
    newpath = '/Users/alexromriell/Dropbox/School/SpringModule1/AdvancedML/Project/pickle/' + subdir
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    pickle_path = "pickle/" + subdir + "/svm.pkl"
    joblib.dump(svm_clf, pickle_path)

    # clf2 = joblib.load(pickle_path)
    # print clf2.score(X_test, y_test)
    # print clf2.predict(...)

    # jacob = '/Users/alexromriell/Documents/DATA/AdvMLData/team_pics/HJACOB1.png'
    # alex = '/Users/alexromriell/Documents/DATA/AdvMLData/team_pics/H0ALEX1.png'
    #
    # X_jacob = io.imread(jacob, as_grey=True)

    # X_alex = io.imread(alex, as_grey=True)
    #
    # X_jacob = transform.resize(X_jacob, [48, 48])
    # X_jacob = ndimage.median_filter(X_jacob, 1.5)

    #
    # X_alex = transform.resize(X_alex, [48, 48])
    # X_alex = ndimage.median_filter(X_alex, 1.5)
    # io.imshow(X_alex)
    # io.show()
    #
    #
    # #
    # # print svm_clf.predict(X_jacob.flatten())
    # print svm_clf.predict(X_alex.flatten())

__author__ = 'jtpollard'

from glob import glob
import numpy as np
import seaborn
import matplotlib.pyplot as plt

# STYLES = plt.style.available

CSVFL = "/Users/jtpollard/MSAN/msan630/Emotobot/csvfiles/large_3072.csv"
SAVEPATH = "/Users/jtpollard/MSAN/msan630/Emotobot/plots/seven_emos_large_3072.png"



def get_array_from_csv(csvfile):
    """
    -read in a csv file containing model diagnostics and return a numpy array
    and the name components for using in the plot titles
    :param csvfile:
    :return:
    """
    model_data = np.genfromtxt(csvfile, delimiter=",", skip_header=1)

    return model_data


if __name__ == "__main__":

    data = get_array_from_csv(CSVFL)

    plt.clf()
    plt.style.use('seaborn-notebook')
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.xlabel('Number of Epochs', fontsize=15)
    plt.ylabel('Cross Entropy Loss', fontsize=15)
    # plt.title('Cross Entropy Loss for Model Classifying\n'
    #           'Faces Expressing {} vs {}'.format(graph_emos[title_emos[i][0]], graph_emos[title_emos[i][1]]))
    epochs = data[:, 0]
    train_loss = data[:, 1]
    test_loss = data[:, 2]
    plt.plot(epochs, train_loss, color="blue", label="Train")
    plt.plot(epochs, test_loss, color="darkorange", label="Test")
    plt.legend(loc="best", prop={'size': 20})
    # savepath = PLOTSDIR + "val_curve_{}_{}.png".format(title_emos[i][0], title_emos[i][1])
    plt.savefig(SAVEPATH, dpi=500, bbox_inches="tight")
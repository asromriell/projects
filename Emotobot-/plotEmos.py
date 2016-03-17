import glob
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn

from collections import defaultdict

if __name__ == '__main__':

    files = glob.glob('/Users/DavidWen/emotobot/log/large_3072_rescale*csv')

    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'surprise', 'unhappy']
    color_map = dict()
    color_map['angry'] = 'red'
    color_map['disgust'] = 'green'
    color_map['fear'] = 'purple'
    color_map['happy'] = 'y'
    color_map['neutral'] = 'black'
    color_map['surprise'] = 'pink'
    color_map['unhappy'] = 'blue'

    # dict for the name of the expression to show on the graph
    graph_emos = dict()
    graph_emos['angry'] = 'Anger'
    graph_emos['disgust'] = 'Disgust'
    graph_emos['fear'] = 'Fear'
    graph_emos['happy'] = 'Happiness'
    graph_emos['neutral'] = 'Neutrality'
    graph_emos['surprise'] = 'Surprise'
    graph_emos['unhappy'] = 'Sadness'

    file_map = defaultdict(list)

    for file in files:
        m = re.search(r"_([a-z]+)_([a-z]+)\.csv", file)
        emo1, emo2 = m.group(1), m.group(2)
        file_map[emo1].append((emo2, file))
        file_map[emo2].append((emo1, file))

    for emo1 in file_map:
        plt.clf()
        plt.style.use('ggplot')
        plt.xlabel('Epoch', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.rcParams['xtick.labelsize'] = 18
        plt.rcParams['ytick.labelsize'] = 18
        #plt.title('Accuracy Classifying Faces Expressing %s vs Other Emotions' % graph_emos[emo1])
        for emo2 in file_map[emo1]:

            accfile = emo2[1]
            comp_emo = emo2[0]
            accs = np.genfromtxt(accfile, delimiter=',')
            accs = np.delete(accs, 0, 0)

            epoch = accs[:, 0]
            accuracy = accs[:, 3]

            line = plt.plot(epoch, accuracy, color=color_map[comp_emo], label=graph_emos[comp_emo])
        plt.legend(prop={'size': 18}, loc=4)
        if emo1 == 'disgust':
            plt.ylim([0.70, 1.00])
        plt.savefig("graph_%s.png" % emo1, dpi=500, bbox_inches="tight")


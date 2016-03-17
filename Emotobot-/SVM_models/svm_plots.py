# __author__ = 'alexromriell'
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-notebook')

path = '/Users/alexromriell/Dropbox/School/SpringModule1/AdvancedML/Project/svm_log.csv'

trial = dict()
cnt = 1
with open(path, 'r') as f:

    for line in f:
        line = line.split(',')
        for i in range(len(line)):
            line[i] = re.sub("\(|\)|\[|\'|\"|\]|\s", '', line[i])  # clean up punctuation but not '.'

        emo1 = line[0]
        emo2 = line[1]
        num_img = int(line[2])
        dim_img = line[3]
        kernel = line[4]
        test_acc = float(line[6])
        train_acc = line[7]
        time = float(line[8])
        loc = line[9]
        trial[cnt] = {'emo1': emo1, 'emo2': emo2, 'num_img': num_img, 'dim_img': dim_img, 'kernel': kernel,
                      'test_acc': test_acc, 'train_acc': train_acc, 'time': time, 'loc': loc}
        cnt += 1

trial = pd.DataFrame.from_dict(trial, orient='index').copy(deep=True)
trial_pert = trial[trial['loc'] == 'perturbed_images/all_images']
trial_disgust = trial_pert[trial['emo1'] == 'D']

# ========================================================================================================
# =====================================          TIMES         ===========================================
# ========================================================================================================

sns.lmplot(x='num_img', y='time', hue='kernel', data=trial, markers=['o', 'x', 'p'],
           size=5, aspect=1.5, legend=False, ci=None, lowess=True)
plt.xlim([-6000, 36000])
plt.tick_params(axis='both', which='major', labelsize=15)
L = plt.legend(prop={'size': 18}, loc=2, markerscale=2)
L.get_texts()[0].set_text('Quadratic')
L.get_texts()[1].set_text('Gaussian')
L.get_texts()[2].set_text('PCA')
plt.ylabel("Runtime (min)", fontsize=15)
plt.xlabel("Number of Images", fontsize=15)
plt.savefig('runtime.png', dpi=500, bbox_inches='tight')

plt.show()

# ========================================================================================================
# =====================================          DISGUST         ===========================================
# ========================================================================================================


fig = plt.figure()
ax = plt.subplot(111)

sns.barplot(x='emo2', y='test_acc', hue='kernel', data=trial_disgust)
plt.ylabel('Test Accuracy (%)', fontsize=15)
plt.xlabel('Emotion', fontsize=15)
plt.tick_params(axis='both', which='major', labelsize=18)
L = plt.legend(prop={'size': 18}, bbox_to_anchor=(1.45, 0.5))
L.get_texts()[0].set_text('Quadratic')
L.get_texts()[1].set_text('Gaussian')
L.get_texts()[2].set_text('PCA')
# # Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.savefig('disgustSVM.png', dpi=500, bbox_inches='tight')
plt.show()

# ========================================================================================================
# =====================================          HAPPY         ===========================================
# ========================================================================================================

path = '/Users/alexromriell/Dropbox/School/SpringModule1/AdvancedML/Project/svm_log2.csv'

trial = dict()
cnt = 1
with open(path, 'r') as f:

    for line in f:
        line = line.split(',')
        for i in range(len(line)):
            line[i] = re.sub("\(|\)|\[|\'|\"|\]|\s", '', line[i])  # clean up punctuation but not '.'
        emo1 = line[0]
        emo2 = line[1]
        num_img = int(line[2])
        dim_img = line[3]
        kernel = line[4]
        test_acc = float(line[6])
        train_acc = line[7]
        time = float(line[8])
        loc = line[9]
        trial[cnt] = {'emo1': emo1, 'emo2': emo2, 'num_img': num_img, 'dim_img': dim_img, 'kernel': kernel,
                      'test_acc': test_acc, 'train_acc': train_acc, 'time': time, 'loc': loc}
        cnt += 1

trial = pd.DataFrame.from_dict(trial, orient='index')
trial_pert = trial[trial['loc'] == 'perturbed_images/all_images']
trial_happy = trial_pert[trial['emo1'] == 'H']
# print trial_happy

fig = plt.figure()
ax = plt.subplot(111)

sns.barplot(x='emo2', y='test_acc', hue='kernel', data=trial_happy)
plt.ylabel('Test Accuracy (%)', fontsize=15)
plt.xlabel('Emotion', fontsize=15)
# plt.title('Happy versus all other emotions')
plt.tick_params(axis='both', which='major', labelsize=18)
L = plt.legend(prop={'size': 18}, bbox_to_anchor=(1.45, 0.5))
L.get_texts()[0].set_text('Quadratic')
L.get_texts()[1].set_text('Gaussian')
L.get_texts()[2].set_text('PCA')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.savefig('happySVM.png', dpi=500, bbox_inches='tight')
plt.show()




#  Emotion Detetion Project: SVM Models
# Students: Alex Romriell, Jacob Pollard, David Wen

from sklearn import decomposition, preprocessing
from skimage import io, transform
from scipy import ndimage
from sklearn import svm
import matplotlib.pyplot as plt
plt.style.use('seaborn-notebook')
import numpy as np


def scree_plot(model, xlim=[-1, 100], ylim=[-0.1, 1], required_var=0.90):
    """Make side-by-side scree plots with a marker for required minimum
    percent variance explained.

    Args:
        model (sklearn.decomposition.pca.PCA): A fitted sklearn
            PCA model.
        required_var (float): A required variance to be plotted
            on the scree plot. Must be between 0 and 1.
        xlim (list): X axis range, e.g. [0, 200].
        ylim (list): Y axis range.
    """
    var = model.explained_variance_ratio_
    # First plot (in subplot)
    plt.figure(figsize=(11, 12))
    fig = plt.subplot(2, 2, 1)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.plot(var, marker='o', linestyle='--')
    plt.xlabel("Component Number")
    plt.ylabel("Proportion of Variance Explained")
    # Second plot
    cumsum = np.cumsum(var)  # Cumulative sum of variances
    plt.subplot(2, 2, 2)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.plot(cumsum, marker='o', linestyle='--')
    # Get number of components to meet required variance
    required_var_components = np.argmax(cumsum >= required_var) + 1
    plt.axvline(x=required_var_components,
                c='r',
                linestyle='dashed',
                label="> {:.0f}% Var. Explained: {} components".format(
                    required_var * 100, required_var_components)
               )
    legend = plt.legend(loc='lower right',
                        frameon=True)
    legend.get_frame().set_facecolor("#FFFFFF")
    plt.xlabel("Number of Components")
    plt.ylabel("Proportion of Variance Explained")
    # Show
    plt.show()

angry = '/Users/alexromriell/Documents/DATA/AdvMLData/all_images/A005305.png'
happy = '/Users/alexromriell/Documents/DATA/AdvMLData/all_images/H003500.png'
surpise = '/Users/alexromriell/Documents/DATA/AdvMLData/all_images/S003782.png'
disgust = '/Users/alexromriell/Documents/DATA/AdvMLData/all_images/D011530.png'

X_angry = io.imread(angry, as_grey=True)
X_happy = io.imread(happy, as_grey=True)
X_surprise = io.imread(surpise, as_grey=True)
X_disgust = io.imread(disgust, as_grey=True)


def pca_projection(X, num_comp):

    # Make the scaler so you can transform your data
    # scaler = preprocessing.StandardScaler().fit(X.astype('float64'))

    # Transform your data using the scaler
    # transformed = scaler.transform(X.astype('float64'))

    pcomp = decomposition.PCA(n_components=num_comp)

    pcomp.fit(X)
    # Project original
    data_reduced = pcomp.transform(X)
    # Transform back to original space (otherwise
    # dimensions don't match orignal image since you
    # only used d components)
    projection = pcomp.inverse_transform(data_reduced)
    # Finally, let's reverse the standard scaler
    # projection = scaler.inverse_transform(projection)
    # Output cumulative percentage variation explained
    var = pcomp.explained_variance_ratio_
    return projection, var

p2_1, var2_1 = pca_projection(X_angry, 2)
p5_1, var5_1 = pca_projection(X_angry, 5)
p15_1, var15_1 = pca_projection(X_angry, 15)
p20_1, var20_1 = pca_projection(X_angry, 20)
p2_2, var2_2 = pca_projection(X_happy, 2)
p5_2, var5_2 = pca_projection(X_happy, 5)
p15_2, var15_2 = pca_projection(X_happy, 15)
p20_2, var20_2 = pca_projection(X_happy, 20)
p2_3, var2_3 = pca_projection(X_surprise, 2)
p5_3, var5_3 = pca_projection(X_surprise, 5)
p15_3, var15_3 = pca_projection(X_surprise, 15)
p20_3, var20_3 = pca_projection(X_surprise, 20)
p2_4, var2_4 = pca_projection(X_disgust, 2)
p5_4, var5_4 = pca_projection(X_disgust, 5)
p15_4, var15_4 = pca_projection(X_disgust, 15)
p20_4, var20_4 = pca_projection(X_disgust, 20)

# Plot the image
# fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(3, 3)
fig, ax = plt.subplots(nrows=4,ncols=4)

ax[0,0].imshow(p2_1, cmap=plt.cm.gray)
ax[0,0].set_title("{} PCs: {:.2f}%".format(2, var2_1.sum() * 100), fontsize=10)
ax[0,0].axis('off')

ax[0,1].imshow(p5_1, cmap=plt.cm.gray)
ax[0,1].set_title("{} PCs: {:.2f}%".format(5, var5_1.sum() * 100), fontsize=10)
ax[0,1].axis('off')

ax[0,2].imshow(p15_1, cmap=plt.cm.gray)
ax[0,2].set_title("{} PCs: {:.2f}%".format(10, var15_1.sum() * 100), fontsize=10)
ax[0,2].axis('off')

ax[0,3].imshow(p20_1, cmap=plt.cm.gray)
ax[0,3].set_title("{} PCs: {:.2f}%".format(20, var20_1.sum() * 100), fontsize=10)
ax[0,3].axis('off')

######

ax[1,0].imshow(p2_2, cmap=plt.cm.gray)
ax[1,0].set_title("{} PCs: {:.2f}%".format(2, var2_2.sum() * 100), fontsize=10)
ax[1,0].axis('off')

ax[1,1].imshow(p5_2, cmap=plt.cm.gray)
ax[1,1].set_title("{} PCs: {:.2f}%".format(5, var5_2.sum() * 100), fontsize=10)
ax[1,1].axis('off')

ax[1,2].imshow(p15_2, cmap=plt.cm.gray)
ax[1,2].set_title("{} PCs: {:.2f}%".format(10, var15_2.sum() * 100), fontsize=10)
ax[1,2].axis('off')

ax[1,3].imshow(p20_2, cmap=plt.cm.gray)
ax[1,3].set_title("{} PCs: {:.2f}%".format(20, var20_2.sum() * 100), fontsize=10)
ax[1,3].axis('off')

####

ax[2,0].imshow(p2_3, cmap=plt.cm.gray)
ax[2,0].set_title("{} PCs: {:.2f}%".format(2, var2_3.sum() * 100), fontsize=10)
ax[2,0].axis('off')

ax[2,1].imshow(p5_3, cmap=plt.cm.gray)
ax[2,1].set_title("{} PCs: {:.2f}%".format(5, var5_3.sum() * 100), fontsize=10)
ax[2,1].axis('off')

ax[2,2].imshow(p15_3, cmap=plt.cm.gray)
ax[2,2].set_title("{} PCs: {:.2f}%".format(10, var15_3.sum() * 100), fontsize=10)
ax[2,2].axis('off')

ax[2,3].imshow(p20_3, cmap=plt.cm.gray)
ax[2,3].set_title("{} PCs: {:.2f}%".format(20, var20_3.sum() * 100), fontsize=10)
ax[2,3].axis('off')

####

ax[3,0].imshow(p2_4, cmap=plt.cm.gray)
ax[3,0].set_title("{} PCs: {:.2f}%".format(2, var2_4.sum() * 100), fontsize=10)
ax[3,0].axis('off')

ax[3,1].imshow(p5_4, cmap=plt.cm.gray)
ax[3,1].set_title("{} PCs: {:.2f}%".format(5, var5_4.sum() * 100), fontsize=10)
ax[3,1].axis('off')

ax[3,2].imshow(p15_4, cmap=plt.cm.gray)
ax[3,2].set_title("{} PCs: {:.2f}%".format(10, var15_4.sum() * 100), fontsize=10)
ax[3,2].axis('off')

ax[3,3].imshow(p20_4, cmap=plt.cm.gray)
ax[3,3].set_title("{} PCs: {:.2f}%".format(20, var20_4.sum() * 100), fontsize=10)
ax[3,3].axis('off')

plt.savefig('pc_faces.png', dpi=500, bbox_inches='tight')
plt.show()


#  io.imshow(X_jacob)
# io.show()


#
# print svm_clf.predict(X_jacob.flatten())
# print svm_clf.predict(X_alex.flatten())
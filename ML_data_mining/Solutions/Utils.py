import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def plot_clusters(X, y, dim, points,
                  labels_prefix = 'cluster', 
                  points_name = 'centroids',
                  colors = cm.tab10, # a qualitative map 
                      # https://matplotlib.org/examples/color/colormaps_reference.html
#                   colors = ['brown', 'orange', 'olive', 
#                             'green', 'cyan', 'blue', 
#                             'purple', 'pink'],
#                   points_color = 'red'
                  points_color = cm.tab10(10) # by default the last of the map (to be improved)
                 ):
    """
    Plot a two dimensional projection of an array of labelled points
    X:      array with at least two columns
    y:      vector of labels, length as number of rows in X
    dim:    the two columns to project, inside range of X columns, e.g. (0,1)
    points: additional points to plot as 'stars'
    labels_prefix: prefix to the labels for the legend ['cluster']
    points_name:   legend name for the additional points ['centroids']
    colors: a color map
    points_color: the color for the points
    """
    # plot the labelled (colored) dataset and the points
    labels = np.unique(y)
    for i in range(len(labels)):
        color = colors(i / len(labels)) # choose a color from the map
        plt.scatter(X[y==labels[i],dim[0]], 
                    X[y==labels[i],dim[1]], 
                    s=10, 
                    c = [color], # scatter requires a sequence of colors
                    marker='s', 
                    label=labels_prefix+str(labels[i]))
    plt.scatter(points[:,dim[0]], 
                points[:,dim[1]], 
                s=50, 
                marker='*', 
                c=[points_color], 
                label=points_name)
    plt.legend()
    plt.grid()
    plt.show()
    
    

def plot_silhouette(silhouette_vals, y, 
 					colors = cm.tab10
					):
    """
    Plotting silhouette scores for the individual samples of a labelled data set.
    The scores will be grouped according to labels and sorted in descending order.
    The bars are proportional to the score and the color is determined by the label.
    
    silhouette_vals: the silhouette values of the samples
    y:               the labels of the samples
    
    """
    cluster_labels = np.unique(y)
    n_clusters = len(cluster_labels)
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    for i, c in enumerate(cluster_labels): # generate pairs index, cluster_label
        c_silhouette_vals = silhouette_vals[y==c] # extracts records with the current cluster label
        c_silhouette_vals.sort() # sort the silhouette vals for the current class
        y_ax_upper += len(c_silhouette_vals)
        color = colors(i / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
                edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)

    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg, color="red", linestyle="--") 
    plt.yticks(yticks, cluster_labels)# + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.tight_layout()
    # plt.savefig('./figures/silhouette.png', dpi=300)
    plt.show()
    
    
    

def max_diag(sq_arr):
    """
    Given a square matrix produces another squared matrix with the same contents, 
    but the columns are re-orered in order to have the highest values in the main diagonal
    Parameter: sq_arr - a squared matrix
    Example:
    In [1]: import numpy as np
            max_diag(np.array([[1,10],[20,2]]))
    Out[1]: array([[10.,  1.],
                   [ 2., 20.]])
    This function is useful to reorder a confusion matrix when the two label vectors
    have different codings
    """
    
    if len(sq_arr.shape) != 2 or sq_arr.shape[0]!=sq_arr.shape[1]:
        return "Not a squared matrix"
    # find the position of the maximum value in each row
    max_pos = [np.argmax(sq_arr[i,:]) for i in range(sq_arr.shape[0])]
    if len(set(max_pos))!=sq_arr.shape[0]:
        return "There are columns with non unique maximum"
    sq_arr_sh = np.empty(sq_arr.shape)
    for i in range(sq_arr.shape[0]):
        sq_arr_sh[:,i] = sq_arr[:,max_pos[i]]
    return(sq_arr_sh)


def Normalization(X):
    Xn = (X-X.min())/(X.max()-X.min())
    Xn = Xn.to_numpy()
    return pd.DataFrame(Xn,columns=X.columns)


def Standardization(X):
    Xs = (X-X.mean())/(X.std())
    Xs = Xs.to_numpy()
    return pd.DataFrame(Xs,columns=X.columns)

def remap(y_true, y_pred, return_map=False):
    """
    Remaps a categorical labeling (such as one predicted by a clustering algorithm) to
    match the labels used by another similar labeling.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels, or, labels to map to.
    y_pred : array-like of shape (n_samples,)
        Labels to remap to match the categorical labeling of `y_true`.

    Returns
    -------
    remapped_y_pred : np.ndarray of shape (n_samples,)
        Same categorical labeling as that of `y_pred`, but with the category labels
        permuted to best match those of `y_true`.
    label_map : dict
        Mapping from the original labels of `y_pred` to the new labels which best
        resemble those of `y_true`.

    Examples
    --------
    >>> y_true = np.array([0,0,1,1,2,2])
    >>> y_pred = np.array([2,2,1,1,0,0])
    >>> remap_labels(y_true, y_pred)
    array([0, 0, 1, 1, 2, 2])

    """
    confusion_mat = confusion_matrix(y_true, y_pred)
    row_inds, col_inds = linear_sum_assignment(confusion_mat, maximize=True)
    label_map = dict(zip(col_inds, row_inds))
    remapped_y_pred = np.vectorize(label_map.get)(y_pred)
    if return_map:
        return remapped_y_pred, label_map
    else:
        return remapped_y_pred
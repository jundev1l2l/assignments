import numpy as np
import matplotlib.pyplot as plt


class Dataset():
    def __init__(self,data,name):
        self.data = data  # [N,2]
        self.name = name


def get_data(is_balanced=True,is_spherical=True,plot=True,num=500,batch_size=100,shuffle=True):
    """
    generate data for each case and plot if plot==True
    -balanced: 500 points for both
    -imbalanced: 500/50 points for each
    -spherical: 2-variable independent normal
    -nonspherical: non-linear functions + random noise
    """
    # generator config
    N1 = num
    if is_balanced:
        N2 = N1
    else:
        N2 = int(N1 / 10)

    # generate data for each case
    if is_spherical:
        data = _get_spherical(N1, N2)
    else:
        data = _get_nonspherical(N1, N2)

    # plot data and save with name
    if plot:
        if is_balanced:
            if is_spherical:
                name = "balanced_spherical"
            else:
                name = "balanced_nonspherical"
        else:
            if is_spherical:
                name = "imbalanced_spherical"
            else:
                name = "imbalanced_nonspherical"
        _plot_data(data, name)

    # combine and return as dataloader
    data = np.concatenate([data[0],data[1]],axis=0)  # [N1+N2,2]
    np.random.shuffle(data)
    dataset = Dataset(data,name)
    return dataset

def _get_spherical(N1, N2):
    # pick 2 means and vars for each clusters
    m1, m2 = np.random.random((2, 2))  # 0~1
    m1 = 3 * m1 + 2  # 2~5
    m2 = 3 * m2 + 5  # 5~8
    v = 0.5  # use same variance for each features, each clusters
    # generate 2 clusters from normal
    c1 = np.random.multivariate_normal(m1, np.diag([v, v]), N1)  # [N1,2]
    c2 = np.random.multivariate_normal(m2, np.diag([v, v]), N2)  # [N2,2]
    return c1, c2  # ([N1,2],[N2,2])

def _get_nonspherical(N1, N2):
    # pick x values + noise
    x1 = np.linspace(0, 1, N1) * 5 + np.random.randn(N1) * 0.1  # [N1,]
    x2 = np.linspace(0, 1, N2) * 5 + np.random.randn(N2) * 0.1 + 5  # [N2,] translation
    # generate half-sin waves + noise
    y1 = np.sin(x1 / 5 * np.pi) + np.random.randn(N1) * 0.1  # [N1,]
    y2 = np.sin(x2 / 5 * np.pi) + np.random.randn(N2) * 0.1  # [N2,]
    # translate and rescale
    x1 = x1 + 1
    x2 = x2 - 1
    y1 = 3 * y1 + 5
    y2 = 3 * y2 + 5
    # return 2 clusters
    c1 = np.stack([x1, y1], axis=-1)  # [N1,2]
    c2 = np.stack([x2, y2], axis=-1)  # [N2,2]
    return c1, c2  # ([N1,2],[N2,2])

def _plot_data(data,name):
    c1, c2 = data

    plt.figure(figsize=(4,4))
    plt.title(name,fontsize=10)
    plt.xlabel("x1",fontsize=10)
    plt.ylabel("x2",fontsize=10)
    plt.scatter(c1[:,0],c1[:,1],10,"r")
    plt.scatter(c2[:,0],c2[:,1],10,"b")

    plt.savefig("artificial/{}.png".format(name))
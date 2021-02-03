from utils import get_data
from model import KMeans

"""
1-A. Generating & Plotting Artificial Data
- "balanced": data with similar number of observations for each class
- "imbalanced": not balanced. data having significantly different amount of observations for classes
- "spherical": distribution of each clusters have spherical shape
- "non-spherical": not spherical
- implemented in utils.py
"""
# data generation
balanced_spherical = get_data(is_balanced=True,is_spherical=True,plot=True,num=500,batch_size=100)
imbalanced_spherical = get_data(is_balanced=False,is_spherical=True,plot=True,num=500,batch_size=100)
balanced_nonspherical = get_data(is_balanced=True,is_spherical=False,plot=True,num=500,batch_size=100)
imbalanced_nonspherical = get_data(is_balanced=False,is_spherical=False,plot=True,num=500,batch_size=100)
"""
1-B. Training K-Means model and Plotting Results
- 
- plot results will be saved in the current folder as png files.
- implemented in main.py
"""
# hyperparameters
K = 2             # number of clusters
Beta = 1.0        # stiffness
epochs = 50
loss_every = 50
# soft-k-means
soft = KMeans(type="soft",num_mix=K,stiffness=Beta)
soft.train(balanced_spherical,epochs,loss_every,plot=True)
soft.train(imbalanced_spherical,epochs,loss_every,plot=True)
soft.train(balanced_nonspherical,epochs,loss_every,plot=True)
soft.train(imbalanced_nonspherical,epochs,loss_every,plot=True)
# hard-k-means
hard = KMeans(type="hard",num_mix=K)
hard.train(balanced_spherical,epochs,loss_every,plot=True)
hard.train(imbalanced_spherical,epochs,loss_every,plot=True)
hard.train(balanced_nonspherical,epochs,loss_every,plot=True)
hard.train(imbalanced_nonspherical,epochs,loss_every,plot=True)


"""
real dataset: 2 img
(1) get images with [W,H,C] (C=3 for rgb) (about 300*400)
(2) each image into H*W data points with C+2 features(2 from x,y axis): [H*W,C+2] 
(3) implement data processing in utils.py
(4) use same algorithms in model.py and run in main.py
"""


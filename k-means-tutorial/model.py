import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


"""
N: num_data
D: num_feat
K: num_mix
"""
class KMeans():
    def __init__(self,type="soft",num_mix=10,stiffness=1.0):  # data:([N1,D=2],[N2,D=2])
        self.type = type
        # hyper-parameter
        self.K = num_mix
        self.beta = stiffness

    def train(self,dataset,epochs=500,loss_every=100,plot=True):
        self.data = dataset.data
        self.N = self.data.shape[0]
        self.name = dataset.name
        self._init()
        print("Start Training: {}_{}".format(self.type,self.name))
        print("Epoch     Loss")
        for epoch in range(epochs):
            self._assign()
            self._update()
            if epoch % loss_every == (loss_every-1):
                print("{:04d}     {:.4f}".format(epoch+1,self._loss()))
        print("Training Finished")
        print()
        if plot:
            self._plot()

    def _init(self):
        """
        random initialization of means
        """
        min = np.min(self.data,axis=0)
        max = np.max(self.data,axis=0)

        self.mean = np.random.uniform(min,max,(self.K,2))  # [K,D=2]

    def _assign(self):
        """
        E-step: assign responsibilities with given means
        """
        data = np.repeat(np.expand_dims(self.data, axis=0), self.K, axis=0)  # [K,N,D=2]
        mean = np.repeat(np.expand_dims(self.mean, axis=1), self.N, axis=1)  # [K,N,D=2]
        if self.type == "soft":
            resp = np.exp(- self.beta * np.sum((self.data - mean)**2,axis=-1))  # [K,N]
            resp_k_sum = np.repeat(np.sum(resp,axis=0,keepdims=True),self.K,axis=0)  # [K,N]
            self.resp = resp / resp_k_sum  # [K,N]
        else:  # "hard"
            idx = np.argmin(np.sum((self.data - mean)**2,axis=-1),axis=0)  # [N]
            self.resp = np.array([[1.0 if k==idx[n] else 0.0 for n in range(self.N)] for k in range(self.K)])  # [K,N]

    def _update(self):
        """
        M-step: update means with given responsibilities
        """
        weighted_mean = np.matmul(self.resp,self.data)  # [K,D=2]
        resp_n_sum = np.sum(self.resp,axis=-1,keepdims=True)  # [K,1]

        self.mean = weighted_mean / resp_n_sum  # [K,D=2]

    def _loss(self):
        """
        return loss for soft-k-means
        """
        data = np.repeat(np.expand_dims(self.data, axis=0), self.K, axis=0)  # [K,N,D=2]
        mean = np.repeat(np.expand_dims(self.mean, axis=1), self.N, axis=1)  # [K,N,D=2]
        exp_energy = np.sum(self.resp * np.sum((data - mean) ** 2, axis=-1))/self.N  # [1,]
        if self.type == "soft":
            entropy = np.sum(self.resp * np.log(self.resp))/self.beta/self.N  #[1,]
            return exp_energy + entropy
        else:  # "hard"
            return exp_energy

    def _plot(self):
        """
        plot data with predicted clusters in different colors
        """
        plt.figure(figsize=(4,4))
        plt.title("{} k-means for {}".format(self.type,self.name))

        # divide data into predicted clusters
        idx = np.argmax(self.resp,axis=0)  # [N,]
        clusters = [self.data[idx==k,:] for k in range(self.K)]  # [K,]

        # plot predicted clusters and means
        for k in range(self.K):
            plt.scatter(self.mean[k,0],self.mean[k,1],marker="*",
                        s=100,alpha=1.0,label="{}".format(k+1))  # k-th mean
            plt.scatter(clusters[k][:,0],clusters[k][:,1],marker=",",
                        s=1,alpha=0.3)  # k-th cluster

        plt.legend(loc="lower left",title="Clusters",fontsize="10")
        plt.savefig("artificial/{}_{}.png".format(self.type,self.name))

from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from sklearn.mixture import BayesianGaussianMixture
import numpy as np

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=256, shuffle=True)


def train(epoch, prior):
    prior = BayesianGaussianMixture(n_components=50, covariance_type='diag', n_init=5, max_iter=1000)
    tmp = []
    for (data,_) in train_loader:
        #print(data.numpy().shape)
        tmp.append(data.numpy().reshape(data.numpy().shape[0],-1))
    prior.fit(np.vstack(tmp))
    return prior


def test(epoch, prior):
    ans = np.zeros((50, 10))
    for data, lab in test_loader:
        C = prior.predict(data.numpy().reshape(data.numpy().shape[0],-1))
        for i in xrange(len(lab)):
            ans[C[i],lab[i]]+=1
    print(ans)
    s = np.sum(ans)
    v = 0
    for i in xrange(ans.shape[0]):
        for j in xrange(ans.shape[1]):
            if ans[i,j]>0:
                v += ans[i,j]/s*np.log(ans[i,j]/s/(np.sum(ans[i,:])/s)/(np.sum(ans[:,j])/s))
    print("Mutual information: "+str(v))

#prior = BayesianGaussianMixture(n_components=100, covariance_type='diag')
for epoch in range(1):
    prior=train(epoch, None)
    test(epoch,prior)


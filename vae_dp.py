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

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--hidden',type=int,default=10,metavar='N',
                    help='number of dimension for z')
parser.add_argument('--comp',type=int,default=100,metavar='N',
                    help='maximum number of components in DP')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, args.hidden)
        self.fc22 = nn.Linear(400, args.hidden)
        self.fc3 = nn.Linear(args.hidden, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def sample(self, model, n):
        z = Variable(torch.from_numpy(model.sample(n)[0].astype(np.float32))).cuda()
        return self.decode(z)


model = VAE()
if args.cuda:
    model.cuda()

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False
C = None
def KL(model, mu, logvar, z):
    global C
    C    = model.predict(z.cpu().data.numpy())
    muc  = Variable(torch.from_numpy(model.means_[C].astype(np.float32)))
    varc = Variable(torch.from_numpy(np.log(model.covariances_[C],dtype=np.float32)))
    if args.cuda:
        muc  = muc.cuda()
        varc = varc.cuda()
    return torch.sum(muc.sub_(mu).pow(2).div(varc.exp()).add_(varc).sub_(logvar).add_(logvar.exp().div(varc.exp()))).mul_(0.5)

def loss_function(recon_x, x, mu, logvar, model, z):
    BCE = reconstruction_function(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    #KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KL(model, mu, logvar, z)


optimizer = optim.Adam(model.parameters(), lr=1e-3)

def getz():
    tmp = []
    for (data,_) in train_loader:
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        recon_batch, mu, logvar, z = model(data)
        tmp.append(z.cpu().data.numpy())
    return np.vstack(tmp)

def train(epoch, prior):
    model.train()
    train_loss = 0
    #prior = BayesianGaussianMixture(n_components=1, covariance_type='diag')
    tmp = []
    for (data,_) in train_loader:
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        recon_batch, mu, logvar, z = model(data)
        tmp.append(z.cpu().data.numpy())
    print('Update Prior')
    prior.fit(np.vstack(tmp))
    print('prior: '+str(prior.weights_))
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, prior, z)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        #if batch_idx % args.log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader),
        #        loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return prior


def test(epoch, prior):
    model.eval()
    test_loss = 0
    ans = np.zeros((args.comp, 10))
    for data, lab in test_loader:
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar, z = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar, prior, z).data[0]
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
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

prior = BayesianGaussianMixture(n_components=args.comp, covariance_type='diag')
for epoch in range(1, args.epochs + 1):
    prior=train(epoch, prior)
    test(epoch,prior)
np.savetxt('z.txt',getz())
regen = model.sample(prior, 1024).cpu().data.numpy()
import cPickle
with open('img.pickle','wb') as f:
    cPickle.dump(regen, f)
np.savetxt('model/fc1_W', model.fc1.weight.cpu().data.numpy())
np.savetxt('model/fc1_b', model.fc1.bias.cpu().data.numpy())
np.savetxt('model/fc21_W', model.fc21.weight.cpu().data.numpy())
np.savetxt('model/fc21_b', model.fc21.bias.cpu().data.numpy())
np.savetxt('model/fc22_W', model.fc22.weight.cpu().data.numpy())
np.savetxt('model/fc22_b', model.fc22.bias.cpu().data.numpy())
np.savetxt('model/fc3_W', model.fc3.weight.cpu().data.numpy())
np.savetxt('model/fc3_b', model.fc3.bias.cpu().data.numpy())
np.savetxt('model/fc4_W', model.fc4.weight.cpu().data.numpy())
np.savetxt('model/fc4_b', model.fc4.bias.cpu().data.numpy())
np.savetxt('model/weights', prior.weights_)
np.savetxt('model/means', prior.means_)
np.savetxt('model/covars', prior.covariances_)

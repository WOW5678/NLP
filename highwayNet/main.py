# -*- coding: utf-8 -*-
"""
 @Time    : 2019/6/10 0010 下午 4:20
 @Author  : Shanshan Wang
 @Version : Python3.5
 @Function:
"""
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets,transforms
from torch.autograd import Variable
from layers.highway import HighwayMLP

# Training settings
parser=argparse.ArgumentParser(description='Pytorch MNIST Example')
#metavar是用在usage说明中的参数名
parser.add_argument('--batch_size',type=int,default=64,metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--highway-number', type=int, default=10, metavar='N',
                    help='how many highway layers to use in the model')
args=parser.parse_args()
args.cuda=not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs={'num_workers':1,'pin_memory':True} if args.cuda else {}

# Load the dataset
train_loader=torch.utils.data.DataLoader(
    datasets.MNIST('../data',train=True,download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Lambda(lambda x:x.numpy().flatten())
                   ])),
    batch_size=args.batch_size,shuffle=True,**kwargs
)
test_loader=torch.utils.data.DataLoader(
    datasets.MNIST('../data',train=False,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x:x.numpy().flatten()),
    ])),
    batch_size=args.batch_size,shuffle=True,**kwargs
)

class Model(nn.Module):
    def __init__(self,input_size,output_size):
        super(Model, self).__init__()
        self.highway_layers=nn.ModuleList([HighwayMLP(input_size,activation_fun=F.relu) for _ in range(args.highway_number)])
        self.linear=nn.Linear(input_size,output_size)

    def forward(self,x):
        for current_layer in self.highway_layers:
            x=current_layer(x)
        x=F.softmax(self.linear(x))
        return x

model=Model(input_size=784,output_size=10)
if args.cuda:
    model.cuda()
optimizer=optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum)

#模型的训练过程
def train(epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        if args.cuda:
            data,target=data.cuda(),target.cuda()
        data,target=Variable(data),Variable(target)
        optimizer.zero_grad()
        output=model(data)
        loss=F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx %args.log_interval==0:
            print('Train Epoch:{}[{}/{} ({:.0f}%)]\t Loss:{:.6f}'.format(
                epoch, batch_idx * len(data),len(train_loader.dataset),
                100. * batch_idx / len(train_loader),loss.item()
            ))

def test(epoch):
    model.eval()
    test_loss=0
    correct=0
    for data,target in test_loader:
        if args.cuda:
            data,target=data.cuda(),target.cuda()

        data,target=Variable(data,volatile=True),Variable(target)

        output=model(data)
        test_loss+=F.nll_loss(output,target).data[0]
        pred=output.data.max(1)[1] # get the index of max log-probability
        correct+=pred.eq(target.data).cpu().sum()

    test_loss=test_loss
    test_loss/=len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#执行训练和测试过程
for epoch in range(1,args.epochs+1):
    print('Train epoch:{}'.format(epoch))
    train(epoch)
    print('Test epoch:{}'.format(epoch))
    test(epoch)
#!/usr/bin/env python
# coding: utf-8
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=str, default='-1', metavar='GPU',
                    help='set GPU id (default: -1)')
parser.add_argument('-b', '--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('-e', '--epochs', type=int, default=100, metavar='E',
                    help='how many epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--decay', type=float, default=1e-5, metavar='D',
                    help='weight decay or L2 penalty (default: 1e-5)')
parser.add_argument('-z', '--zdim', type=int, default=128, metavar='Z',
                    help='dimension of latent vector (default: 128)')

opt = parser.parse_args()

# set params
# ===============
cuda = 0 if opt.gpu == -1 else 1
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
BS = opt.batch_size
Zdim = opt.zdim
IMAGE_PATH = 'images'
MODEL_PATH = 'models'

# ===============
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from itertools import chain
from torchvision.utils import save_image
from model import *
from utils import *

os.chdir('..')
if not os.path.exists(IMAGE_PATH):
    print('mkdir ', IMAGE_PATH)
    os.mkdir(IMAGE_PATH)
if not os.path.exists(MODEL_PATH):
    print('mkdir ', MODEL_PATH)
    os.mkdir(MODEL_PATH)


def train():
    # load models
    Gx = GeneratorX(zd=Zdim)
    Gz = GeneratorZ(zd=Zdim)
    Dx = DiscriminatorX(zd=Zdim)
    Dxz = DiscriminatorXZ(zd=Zdim)

    # load dataset
    # ==========================
    kwargs = dict(num_workers=1, pin_memory=True) if cuda else {}
    dataloader = DataLoader(
        datasets.MNIST('MNIST', download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=BS, shuffle=True, **kwargs
    )
    N = len(dataloader)

    z = torch.FloatTensor(BS, Zdim, 1, 1).normal_(0, 1)
    z_pred = torch.FloatTensor(81, Zdim, 1, 1).normal_(0, 1)
    z_pred = Variable(z_pred)
    noise = torch.FloatTensor(BS, Zdim, 1, 1).normal_(0, 1)

    if cuda:
        Gx.cuda()
        Gz.cuda()
        Dx.cuda()
        Dxz.cuda()
        z, z_pred, noise = z.cuda(), z_pred.cuda(), noise.cuda()


    # optimizer
    optim_g = optim.Adam(chain(Gx.parameters(),Gz.parameters()),
                         lr=opt.lr, betas=(.5, .999), weight_decay=opt.decay)
    optim_d = optim.Adam(chain(Dx.parameters(),Dxz.parameters()),
                         lr=opt.lr, betas=(.5, .999), weight_decay=opt.decay)

    # train
    # ==========================
    softplus = nn.Softplus()
    d_interval = 3
    for epoch in range(opt.epochs):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            if cuda:
                imgs = imgs.cuda()
            imgs = Variable(imgs)
            z.resize_(batch_size, Zdim, 1, 1).normal_(0, 1)
            noise.resize_(batch_size, Zdim, 1, 1).normal_(0, 1)
            zv = Variable(z)
            noisev = Variable(noise)

            # forward
            imgs_fake = Gx(zv)
            encoded = Gz(imgs)
            # reparametrization trick
            z_enc = encoded[:, :Zdim] + encoded[:, Zdim:].exp() * noisev
            dx_true = Dx(imgs)
            dx_fake = Dx(imgs_fake)
            d_true = Dxz(torch.cat((dx_true, z_enc), dim=1))
            d_fake = Dxz(torch.cat((dx_fake, zv), dim=1))

            # compute loss
            loss_d = torch.mean(softplus(-d_true) + softplus(d_fake))
            loss_g = torch.mean(softplus(d_true) + softplus(-d_fake))
            if i % d_interval == 0:  # update D
                # for p in Dx.parameters():
                #     p.requires_grad = True
                # for p in Dxz.parameters():
                #     p.requires_grad = True
                # for p in Gx.parameters():
                #     p.requires_grad = False
                # for p in Gz.parameters():
                #     p.requires_grad = False
                # backward & update params
                Dx.zero_grad()
                Dxz.zero_grad()
                loss_d.backward()
                optim_d.step()
            else:  # update G
                # for p in Dx.parameters():
                #     p.requires_grad = False
                # for p in Dxz.parameters():
                #     p.requires_grad = False
                # for p in Gx.parameters():
                #     p.requires_grad = True
                # for p in Gz.parameters():
                #     p.requires_grad = True
                Gx.zero_grad()
                Gz.zero_grad()
                loss_g.backward()
                optim_g.step()

            prog_ali(epoch+1, i+1, N, loss_g.data[0], loss_d.data[0], d_true.data.mean(), d_fake.data.mean())

        # generate fake images
        save_image(Gx(z_pred).data,
                   os.path.join(IMAGE_PATH,'%d.png' % (epoch+1)),
                   nrow=9, padding=1,
                   normalize=False)
        # save models
        torch.save(Gx.state_dict(),
                   os.path.join(MODEL_PATH, 'Gx-%d.pth' % (epoch+1)))
        torch.save(Gz.state_dict(),
                   os.path.join(MODEL_PATH, 'Gz-%d.pth' % (epoch+1)))
        torch.save(Dx.state_dict(),
                   os.path.join(MODEL_PATH, 'Dx-%d.pth'  % (epoch+1)))
        torch.save(Dxz.state_dict(),
                   os.path.join(MODEL_PATH, 'Dxz-%d.pth'  % (epoch+1)))
        print()


if __name__ == '__main__':
    train()

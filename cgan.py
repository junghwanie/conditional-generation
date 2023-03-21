from __future__ import print_function
import argparse
import enum
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np

img_save_path = './'

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-f')

opt = parser.parse_args()

C,H,W = 1, opt.imageSize, opt.imageSize

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  
dataroot = './'
dataset = dset.MNIST(root=dataroot, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(opt.imageSize),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,)),
                       ]))
nc=1

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, drop_last=True, num_workers=0)

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)
nz = int(opt.nz)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
        
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1_1 = nn.Linear(100, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(10, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, H*W)
        
    def forward(self, z, labels):
        x = F.relu(self.fc1_1_bn(self.fc1_1(z)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(labels)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = torch.tanh(self.fc4(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1_1 = nn.Linear(H*W, 1024)
        self.fc1_2 = nn.Linear(10, 1024)
        self.fc2 = nn.Linear(2048, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)
    
    def forward(self, x, labels):
        x = F.leaky_relu(self.fc1_1(x.view(x.size(0), -1)), 0.2)
        y = F.leaky_relu(self.fc1_2(labels), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = torch.sigmoid(self.fc4(x))
        return x
    
netG = Generator()
netD = Discriminator()

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# Start training
for epoch in range(opt.niter):
    for i, (imgs, labels) in enumerate(dataloader):
        Batch_size = opt.batchSize
        N_Class = 10
        valid = Variable(torch.ones(Batch_size), requires_grad=False)
        fake = Variable(torch.zeros(Batch_size), requires_grad=False)
        
        real_imgs = Variable(imgs.type(torch.FloatTensor))
        # one-hot vector of labels for real image
        real_y = torch.zeros(Batch_size, N_Class)
        real_y = Variable(real_y.scatter_(1, labels.view(Batch_size, 1), 1))
        
        noise = Variable(torch.randn((Batch_size, nz)))
        # one-hot vector of labels for fake image
        gen_labels = (torch.rand(Batch_size, 1) * N_Class).type(torch.LongTensor)
        gen_y = torch.zeros(Batch_size, N_Class)
        gen_y = Variable(gen_y.scatter_(1, gen_labels.view(Batch_size, 1), 1))
        
        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #
        
        optimizerD.zero_grad()
        
        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        real_validity = netD(real_imgs, real_y).squeeze()
        d_real_loss = criterion(real_validity, valid)
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        gen_imgs = netG(noise, gen_y)
        fake_validity_1 = netD(gen_imgs.detach(), gen_y).squeeze()
        d_fake_loss = criterion(fake_validity_1, fake)
        d_loss = (d_real_loss + d_fake_loss)
    
        d_loss.backward()
        D_xy = real_validity.mean().item()
        D_G_zy1 = fake_validity_1.mean().item()
        optimizerD.step()
        
        # ================================================================== #
        #                           Train the generator                      #
        # ================================================================== #
        
        optimizerG.zero_grad()
        fake_validity_2 = netD(gen_imgs, gen_y).squeeze()
        g_loss = criterion(fake_validity_2, valid)
        
        g_loss.backward()
        D_G_zy2 = fake_validity_2.mean().item()
        optimizerG.step()
        
        if (i+1) % 200 == 0:
            print("[%d/%d][%d/%d] loss_D: %.4f loss_G: %.4f D(x|y) %.4f D(G(z|y) %.4f / %.4f" % (epoch, opt.niter, i, len(dataloader),
                    d_loss.data.cpu(), g_loss.data.cpu(), D_xy, D_G_zy1, D_G_zy2))

        batches_done = epoch * len(dataloader) + i
        if batches_done % 200 == 0:
            noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (N_Class**2, nz))))
            #fixed labels
            y_ = torch.LongTensor(np.array([num for num in range(N_Class)])).view(N_Class,1).expand(-1,N_Class).contiguous()
            y_fixed = torch.zeros(N_Class**2, N_Class)
            y_fixed = Variable(y_fixed.scatter_(1,y_.view(N_Class**2,1),1))

            gen_imgs = netG(noise, y_fixed).view(-1,C,H,W)

            save_image(gen_imgs.data, img_save_path + '/%d-%d.png' % (epoch,batches_done), nrow=N_Class, normalize=True)
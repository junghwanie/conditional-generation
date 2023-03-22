import os
import logging
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = "cuda:0" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger()
logger.info(f'Using DEVICE: {device}')


""" path directory should be fixed by users """
path_ls = ['/Users', 'ijeonghwan', 'miniforge3', 'pytorch', 'pix2pix', 'pix2pix-dataset']
path_rt = os.path.join(*path_ls)
pix2pix_filename = 'facades'

"""
if len(sys.argv) != 2:
    print("Usage : python3 dataset.py <pix2pix-dataset-four_categories>")
else:
    pix2pix_filename = sys.argv[1]
"""

data_dir = os.path.join(path_rt, pix2pix_filename, pix2pix_filename)
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

class Pix2PixDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_fns = os.listdir(img_dir)
        logger.info(f'Loaded {len(self)} files')
    
    def __len__(self):
        return len(self.img_fns)

    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        img_fp = os.path.join(self.img_dir, img_fn)
        image = Image.open(img_fp).convert("RGB")
        image = np.array(image)
        img_trg, img_src = self.split_image(image)
        #img_src = self.transform(img_src)
        #img_trg = self.transform(img_trg)
        return img_trg, img_src

    def split_image(self, image):
        image = np.array(image)
        target, source = image[:, :256, :], image[:, 256:, :]
        return target, source

    def transform(self, image):
        transform_ops = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return transform_ops(image)

    def choose(self): return self[np.random.randint(len(self))]

    def collate_fn(self, batch):
        trgs, srcs = list(zip(*batch))
        trgs = torch.cat([self.transform(img)[None] for img in trgs], 0).to(device).float()
        srcs = torch.cat([self.transform(img)[None] for img in srcs], 0).to(device).float()
        return trgs.to("cpu"), srcs.to("cpu")

"""
dataset = Pix2PixDataset(train_dir)
image_source, image_target = dataset[0]
image_source, image_target = Image.fromarray(image_source), Image.fromarray(image_target)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image_source)
axes[1].imshow(image_target)
plt.show()
"""

trn_ds = Pix2PixDataset(train_dir)
trn_dl = DataLoader(trn_ds, batch_size=32, shuffle=True, collate_fn=trn_ds.collate_fn)

def init_weights_mormal(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight'):
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

class UnetDown(nn.Module):
    def __init__(self, input_nc, enc_nc, norm_layer=True, use_dropout=True):
        super(UnetDown, self).__init__()
        downconv = [nn.Conv2d(input_nc, enc_nc, kernel_size=4, 
        stride=2, padding=1, bias=False)]
        if norm_layer:
            downconv.append(nn.InstanceNorm2d(enc_nc))
        downconv.append(nn.LeakyReLU(0.2))
        if use_dropout:
            downconv.append(nn.Dropout(0.5))
        
        self.model = nn.Sequential(*downconv)
    
    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, input_nc, dec_nc, use_dropout=0.0):
        super(UnetUp, self).__init__()
        upconv = [
            nn.ConvTranspose2d(input_nc, dec_nc, kernel_size=4, 
        stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(dec_nc),
            nn.ReLU(inplace=True),]
        if use_dropout:
            upconv.append(nn.Dropout(use_dropout))
        
        self.model = nn.Sequential(*upconv)

    def forward(self, x, skip):
        x = self.model(x)
        x = torch.cat((x, skip), 1)

        return x
        
class UnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(UnetGenerator, self).__init__()
        # Construct unet structure
        self.unet_enc_1 = UnetDown(input_nc, ngf, norm_layer=False, use_dropout=False)
        self.unet_enc_2 = UnetDown(ngf, ngf*2)
        self.unet_enc_3 = UnetDown(ngf*2, ngf*4)
        self.unet_enc_4 = UnetDown(ngf*4, ngf*8, use_dropout=0.5)
        self.unet_enc_5 = UnetDown(ngf*8, ngf*8, use_dropout=0.5) # 512 x 4 layer
        self.unet_enc_6 = UnetDown(ngf*8, ngf*8, use_dropout=0.5)
        self.unet_enc_7 = UnetDown(ngf*8, ngf*8, use_dropout=0.5)
        self.unet_enc_8 = UnetDown(ngf*8, ngf*8, norm_layer=False, use_dropout=0.5)

        self.unet_dec_1 = UnetUp(ngf*8, ngf*8, use_dropout=0.5)
        self.unet_dec_2 = UnetUp(ngf*16, ngf*8, use_dropout=0.5)
        self.unet_dec_3 = UnetUp(ngf*16, ngf*8, use_dropout=0.5)
        self.unet_dec_4 = UnetUp(ngf*16, ngf*8, use_dropout=0.5)
        self.unet_dec_5 = UnetUp(ngf*16, ngf*4)
        self.unet_dec_6 = UnetUp(ngf*8, ngf*2)
        self.unet_dec_7 = UnetUp(ngf*4, ngf)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(ngf*2, output_nc, kernel_size=4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        encoder_1_out = self.unet_enc_1(x)
        encoder_2_out = self.unet_enc_2(encoder_1_out)
        encoder_3_out = self.unet_enc_3(encoder_2_out)
        encoder_4_out = self.unet_enc_4(encoder_3_out)
        encoder_5_out = self.unet_enc_5(encoder_4_out)
        encoder_6_out = self.unet_enc_6(encoder_5_out)
        encoder_7_out = self.unet_enc_7(encoder_6_out)
        encoder_8_out = self.unet_enc_8(encoder_7_out)

        decoder_1_out = self.unet_dec_1(encoder_8_out, encoder_7_out)
        decoder_2_out = self.unet_dec_2(decoder_1_out, encoder_6_out)
        decoder_3_out = self.unet_dec_3(decoder_2_out, encoder_5_out)
        decoder_4_out = self.unet_dec_4(decoder_3_out, encoder_4_out)
        decoder_5_out = self.unet_dec_5(decoder_4_out, encoder_3_out)
        decoder_6_out = self.unet_dec_6(decoder_5_out, encoder_2_out)
        decoder_7_out = self.unet_dec_7(decoder_6_out, encoder_1_out)
        
        return self.final(decoder_7_out)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super(Discriminator, self).__init__()

        def discriminator_block(inner_nc, outer_nc, norm_layer=True):
            disconv = [nn.Conv2d(inner_nc, outer_nc, kernel_size=4, 
            stride=2, padding=1)]
            if norm_layer:
                disconv.append(nn.InstanceNorm2d(outer_nc))
            disconv.append(nn.LeakyReLU(0.2))
            return disconv

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, ndf, norm_layer=False),
            *discriminator_block(ndf, ndf*2),
            *discriminator_block(ndf*2, ndf*4),
            *discriminator_block(ndf*4, ndf*8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, 
            padding=1, bias=False) 
        )

    def forward(self, img_src, img_trg):
        img_input = torch.cat((img_src, img_trg), 1)
        return self.model(img_input)

generator = UnetGenerator().to("cpu")
discriminator = Discriminator().to("cpu")
print(generator)
print(discriminator)

generator.apply(init_weights_mormal)
discriminator.apply(init_weights_mormal)

criterion_GAN = nn.MSELoss()
criterion_pixelwise = nn.L1Loss()

lambda_pixel = 100
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

val_ds = Pix2PixDataset(val_dir)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=val_ds.collate_fn)

def discriminator_train_step(real_src, real_trg, fake_trg):
    discriminator.train()
    d_optimizer.zero_grad()

    prediction_real = discriminator(real_trg, real_src)
    error_real = criterion_GAN(prediction_real, torch.ones(len(real_src), 1, 16, 16))
    error_real.backward()

    prediction_fake = discriminator(fake_trg.detach(), real_src)
    error_fake = criterion_GAN(prediction_fake, torch.zeros(len(real_src), 1, 16, 16))
    error_fake.backward()

    d_optimizer.step()

    return error_real + error_fake, prediction_real, prediction_fake

def generator_train_step(real_src, fake_trg):
    discriminator.train()
    g_optimizer.zero_grad()
    prediction = discriminator(fake_trg, real_src)

    loss_GAN = criterion_GAN(prediction, torch.ones(len(real_src), 1, 16, 16))
    loss_pixel = criterion_pixelwise(fake_trg, real_trg)
    loss_G = loss_GAN + lambda_pixel * loss_pixel

    loss_G.backward()
    g_optimizer.step()
    return loss_G

denorm = transforms.Normalize((-1, -1, -1), (2, 2, 2))
def sample_prediction():
    """Saves a generated sample from the validation set"""
    data = next(iter(val_dl))
    real_src, real_trg = data
    fake_trg = generator(real_src)

    img_sample = torch.cat([denorm(real_src[0]), denorm(fake_trg[0]), denorm(real_trg[0])], -1)
    img_sample = img_sample.detach().cpu().permute(1,2,0).numpy()
    
    plt.imshow(img_sample)
    plt.title("GroundTruth::Generated::Source")
    
    plt.show()

epochs = 100
step_losses = []
epoch_losses = []
for epoch in range(epochs):
    epoch_loss = 0
    for X, Y in enumerate(trn_dl):
        real_src, real_trg = Y
        fake_trg = generator(real_src)
        errD, d_loss_real, d_loss_fake = discriminator_train_step(real_src, real_trg, fake_trg)
        errG = generator_train_step(real_src, fake_trg)

        print('EPOCH: {} Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}'.format(
            epoch, errD.item(), errG.item(), d_loss_real.mean().item(), d_loss_fake.mean().item()))
        [sample_prediction() for _ in range(2)]
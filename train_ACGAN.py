import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
import torch.nn as nn
from IPython import display
import random
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torch
from dataset import faceDatasetACGAN
import ACGAN_model
from utils import Logger
import numpy as np

def real_data_target(size, d_flag):

    if d_flag:
        if random.random() < 0.15:
            data = Variable(torch.FloatTensor(size,1).uniform_(0.0, 0.1))
        else:
            data = Variable(torch.FloatTensor(size,1).uniform_(0.9, 1.0))
    else:
        data = Variable(torch.ones(size,1))

    return data.cuda()

def fake_data_target(size, d_flag):

    if d_flag:
        if random.random() < 0.15:
            data = Variable(torch.FloatTensor(size,1).uniform_(0.9, 1.0))
        else:
            data = Variable(torch.FloatTensor(size,1).uniform_(0.0, 0.1))
    else:
        data = Variable(torch.zeros(size,1))

    return data.cuda()

if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    print(use_gpu)

    # Hyperparm
    file_root = 'hw3_data/face/train/'
    file = 'hw3_data/face/train.csv'
    loadG= torch.load('models/G_ACGAN')
    loadD = torch.load('models/D_ACGAN')

    batch_size = 128
    img_size = 64
    n_epochs = 100

    generator = ACGAN_model.Generator()
    generator.apply(ACGAN_model.init_weights)

    discriminator = ACGAN_model.Discriminator()
    discriminator.apply(ACGAN_model.init_weights)

    if use_gpu:
        generator.cuda()
        discriminator.cuda()


    # Crete dataset
    train_dataset = faceDatasetACGAN(root=file_root, file=file, img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    num_batches = train_loader.__len__()

    # Optimizer
    d_opt = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_opt = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    #Loss
    loss_rf = nn.BCELoss()
    loss_cl = nn.NLLLoss()

    # Laod values
    generator.load_state_dict(loadG['generator_state_dict'])
    g_opt.load_state_dict(loadG['generator_opt_state_dict'])

    discriminator.load_state_dict(loadD['discriminator_state_dict'])
    d_opt.load_state_dict(loadD['discriminator_opt_state_dict'])

    # Test
    num_test_samples = 16
    cl = np.random.randint(0, 2, 16)
    noise_ = np.random.normal(0, 1, (16, 128))
    class_onehot = np.zeros((16, 2))
    class_onehot[np.arange(16), cl] = 1
    noise_[np.arange(16), :2] = class_onehot[np.arange(16)]
    test_noise = (torch.from_numpy(noise_).float()).cuda()

    # Training
    logger = Logger(model_name='ACGAN', data_name='face')
    for epoch in range(n_epochs):
        for i, (real_imgs, cl_real_label ) in enumerate(train_loader):

            real_imgs = Variable(real_imgs).cuda()
            cl_real_label = Variable(cl_real_label ).cuda()

            # Train D wiht real data
            d_opt.zero_grad()

            rf_real_out, cl_real_out = discriminator(real_imgs)
            errorD_real_rf = loss_rf(rf_real_out, real_data_target(real_imgs.size(0), d_flag=True))
            errorD_real_cl = loss_cl(cl_real_out, torch.max(cl_real_label, 1)[1])
            errorD_real = errorD_real_cl + errorD_real_rf
            errorD_real.backward()
            D_x = rf_real_out.data.mean()

            # Train D with fake data
            cl = np.random.randint(0, 2, real_imgs.size(0))
            noise_ = np.random.normal(0, 1, (real_imgs.size(0), 128))
            class_onehot = np.zeros((real_imgs.size(0), 2))
            class_onehot[np.arange(real_imgs.size(0)), cl] = 1
            noise_[np.arange(real_imgs.size(0)), :2] = class_onehot[np.arange(real_imgs.size(0))]
            noise_ = (torch.from_numpy(noise_).float()).cuda()

            fake_data = generator(noise_)
            rf_fake_out, cl_fake_out = discriminator(fake_data.detach())
            errorD_fake_rf = loss_rf(rf_fake_out, fake_data_target(real_imgs.size(0), d_flag=True))
            errorD_fake_cl = loss_cl(cl_fake_out, torch.from_numpy(cl).cuda())
            errorD_fake = errorD_fake_cl + errorD_fake_rf
            errorD_fake.backward()
            D_G_z = rf_fake_out.data.mean()

            errorD = errorD_real + errorD_fake

            d_opt.step()

            ########################################################################################
            # Train G
            g_opt.zero_grad()

            cl = np.random.randint(0, 2, real_imgs.size(0))
            noise_ = np.random.normal(0, 1, (real_imgs.size(0), 128))
            class_onehot = np.zeros((real_imgs.size(0), 2))
            class_onehot[np.arange(real_imgs.size(0)), cl] = 1
            noise_[np.arange(real_imgs.size(0)), :2] = class_onehot[np.arange(real_imgs.size(0))]
            noise_ = (torch.from_numpy(noise_).float()).cuda()

            fake_data = generator(noise_)

            rf_fake_out, cl_fake_out = discriminator(fake_data)
            errorG_rf = loss_rf(rf_fake_out, real_data_target(rf_fake_out.size(0), d_flag=False))
            errorG_cl = loss_cl(cl_fake_out, torch.from_numpy(cl).cuda())
            errorG = errorG_cl + errorG_rf
            errorG.backward()

            g_opt.step()


            # Display Progress
            if i == 0:
                display.clear_output(True)
                # Display Images
                test_images = generator(test_noise).data.cpu()
                logger.log_images(test_images, num_test_samples, epoch, i, num_batches);
                # Display status Logs
                logger.display_status(
                    epoch, n_epochs, i, num_batches,
                    errorD, errorG, D_x, D_G_z
                )
            # Model Checkpoints
        if (epoch % 5 == 0) or (epoch == n_epochs-1):
            logger.save_models(generator, discriminator, g_opt, d_opt, epoch)
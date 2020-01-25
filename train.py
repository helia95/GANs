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
from dataset import faceDataset
import GAN_model
from utils import Logger

def train_discriminator(discriminator, loss, optimizer, real_data, fake_data):

    optimizer.zero_grad()

    prediction_real = discriminator(real_data.cuda())
    error_real = loss(prediction_real, real_data_target(real_data.size(0), d_flag=True))
    error_real.backward()

    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, fake_data_target(real_data.size(0), d_flag=True))
    error_fake.backward()


    optimizer.step()


    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(discriminator, loss, optimizer, fake_data):

    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    error = loss(prediction, real_data_target(prediction.size(0), d_flag=False))
    error.backward()
    optimizer.step()

    return error

def real_data_target(size, d_flag):

    if d_flag:
        if random.random() < 0.15:
            data = Variable(torch.FloatTensor(size,1).uniform_(0.0, 0.1))
        else:
            data = Variable(torch.FloatTensor(size,1).uniform_(0.9, 1.0))
    else:
        data = Variable(torch.ones(size,1))
    if torch.cuda.is_available(): return data.cuda()
    return data

def fake_data_target(size, d_flag):

    if d_flag:
        if random.random() < 0.15:
            data = Variable(torch.FloatTensor(size,1).uniform_(0.9, 1.0))
        else:
            data = Variable(torch.FloatTensor(size,1).uniform_(0.0, 0.1))
    else:
        data = Variable(torch.zeros(size))
    if torch.cuda.is_available(): return data.cuda()
    return data

if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    print(use_gpu)

    # Hyperparm
    file_root = 'hw3_data/face/train/'
    log_d = torch.load('last/D_epoch_50')
    log_g = torch.load('last/G_epoch_50')
    batch_size = 128
    img_size = 64
    n_epochs = 51

    generator = GAN_model.Generator()
    generator.apply(GAN_model.init_weights)

    discriminator = GAN_model.Discriminator()
    discriminator.apply(GAN_model.init_weights)

    if use_gpu:
        generator.cuda()
        discriminator.cuda()


    # Crete dataset
    train_dataset = faceDataset(root=file_root, img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    num_batches = train_loader.__len__()
    # Optimizer
    d_opt = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_opt = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    #Loss
    loss = nn.BCELoss()


    # Load values
    generator.load_state_dict(log_g['generator_state_dict'])
    g_opt.load_state_dict(log_g['generator_opt_state_dict'])

    discriminator.load_state_dict(log_d['discriminator_state_dict'])
    d_opt.load_state_dict(log_d['discriminator_opt_state_dict'])

    # Test
    num_test_samples = 16
    test_noise = GAN_model.noise(num_test_samples)

    # Training
    logger = Logger(model_name='DCGAN', data_name='face')
    for epoch in range(n_epochs):
        for i, real_imgs in enumerate(train_loader):

            real_data = Variable(real_imgs)
            if use_gpu:
                real_data.cuda()
            fake_data = generator(GAN_model.noise(real_data.size(0)).detach())

            # Train D
            d_error, d_pred_real, d_pred_fake = train_discriminator(discriminator, loss, d_opt, real_data, fake_data)

            fake_data = generator(GAN_model.noise(real_data.size(0)))
            g_error = train_generator(discriminator, loss, g_opt, fake_data)

            # Log error
            logger.log(d_error, g_error, epoch, i, num_batches)

            # Display Progress
            if i == 0:
                display.clear_output(True)
                # Display Images
                test_images = generator(test_noise).data.cpu()
                logger.log_images(test_images, num_test_samples, epoch, i, num_batches);
                # Display status Logs
                logger.display_status(
                    epoch, n_epochs, i, num_batches,
                    d_error, g_error, d_pred_real, d_pred_fake
                )
            # Model Checkpoints
        if epoch % 5 == 0:
            logger.save_models(generator, discriminator, g_opt, d_opt, epoch)

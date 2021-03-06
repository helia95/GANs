import torch.nn as nn
import torch
from torch.autograd import Variable



class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.conv3 = nn.Sequential(

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.conv4 = nn.Sequential(

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.conv5 = nn.Sequential(

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.conv6 = nn.Sequential(

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.out = nn.Sequential(

            nn.Linear(8*8*512, 1),
            nn.Sigmoid()

        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 512*8*8)
        x = self.out(x)

        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(128, 1024)

        self.convt1 = nn.Sequential(

            nn.ConvTranspose2d(in_channels=1024, out_channels=512,kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.2, inplace=False)
        )

        self.convt2 = nn.Sequential(

            nn.ConvTranspose2d(in_channels=512, out_channels=256,kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Dropout(0.2, inplace=False)
        )

        self.convt3 = nn.Sequential(

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.2, inplace=False)
        )

        self.convt4 = nn.Sequential(

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

        )

        self.out = nn.Sequential(

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )


    def forward(self, x):
        x = x.view(-1, 128)
        x = self.fc1(x)
        x = x.view(-1, 1024, 1, 1)
        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)
        x = self.convt4(x)
        x = self.out(x)

        return x

def noise(size):
    n = Variable(torch.randn(size, 128))
    if torch.cuda.is_available(): return n.cuda()
    return n

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



if __name__ == '__main__':

    import torch
    import dataset
    from torch.utils.data import DataLoader
    img_size = 64
    batch_size = 1

    file_root = 'hw3_data/face/train/'
    train_dataset = dataset.faceDataset(root=file_root, img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    n = iter(train_loader)



    G = Generator()
    D = Discriminator()

    G.cuda()
    D.cuda()

    test_d = Variable(next(n))
    test_g = Variable(noise(1))
    test_d.cuda()

    out_g = G(test_g)
    out_d = D(test_d)

    print(out_d.shape)
    print(out_g.shape)
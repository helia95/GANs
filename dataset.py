import os
import os.path
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import csv
import numpy as np

class faceDataset(data.Dataset):

    def __init__(self, root, img_size):
        self.root = root
        self.image_size = img_size
        self.transform = transforms.Compose([

            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))

        ])

        self.fnames = os.listdir(root)

        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root + fname))
        img = self.transform(img)

        return img


    def __len__(self):
        return self.num_samples

class faceDatasetACGAN(data.Dataset):

    def __init__(self, root, file, img_size):
        self.root = root
        self.file = file
        self.image_size = img_size
        self.classes = []
        self.transform = transforms.Compose([

            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))

        ])

        self.fnames = os.listdir(root)
        self.num_samples = len(self.fnames)

        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    class_onehot = np.zeros((1, 2))
                    class_onehot[np.arange(1), int(float(row[10]))] = 1

                    self.classes.append( torch.LongTensor( class_onehot ) )
                    line_count += 1
                else:
                    line_count += 1

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        t = self.classes[idx]
        img = Image.open(os.path.join(self.root + fname))
        img = self.transform(img)


        return img, t.squeeze(0)


    def __len__(self):
        return self.num_samples

class DANN(data.Dataset):
    def __init__(self, name, img_size,train):
        self.name = name
        self.labels = []
        self.fnames = []
        self.train = train
        self.image_size = img_size
        self.transform = transforms.Compose([

            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))

        ])

        self.root = 'hw3_data/digits/{}'.format(self.name)
        self.file = 'hw3_data/digits/{}'.format(self.name)
        if self.train:
            self.root = self.root + '/train/'
            self.file = self.file + '/train.csv'
        else:
            self.root = self.root + '/test/'
            self.file = self.file + '/test.csv'


        with open(self.file) as f:
            csv_reader = csv.reader(f, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    self.fnames.append(row[0])
                    self.labels.append(int(row[1]))
                    line_count += 1
                else:
                    line_count += 1

        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        l = self.labels[idx]
        img = Image.open(os.path.join(self.root + fname)).convert('RGB')
        img = self.transform(img)

        if self.train:
            return img, l
        else:
            return img, l, fname

    def __len__(self):
        return self.num_samples




class DANN_test(data.Dataset):
    def __init__(self, root, img_size):
        self.image_size = img_size
        self.transform = transforms.Compose([

            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))

        ])

        self.root = root
        self.fnames = os.listdir(root)
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join('{}/{}'.format(self.root,fname))).convert('RGB')
        img = self.transform(img)

        return img, fname

    def __len__(self):
        return self.num_samples
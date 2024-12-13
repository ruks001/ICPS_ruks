import os
import cv2
import sys
import torch
import glob
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.autograd import Variable
import torch.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from Unet import UNet
import matplotlib.pyplot as plt


def gt_filter(image):
    bg, class1, class2, class3 = image.copy(), image.copy(), image.copy(), image.copy()
    class1[class1!=1]=0

    class2[class2!=2]=0
    class2[class2!=0]=1

    class3[class3 != 3] = 0
    class3[class3 != 0] = 1

    bg[bg!=0]=1
    bg = 1 - bg

    return np.concatenate((np.expand_dims(bg, axis=0), np.expand_dims(class1, axis=0), np.expand_dims(class2, axis=0), np.expand_dims(class3, axis=0)), axis=0)


def dice_similarity_coefficient(y_true, y_predicted, axis=(2,3), epsilon=0.00001):
    dice_numerator = 2. * torch.sum(y_true * y_predicted, dim=axis) + epsilon
    dice_denominator = torch.sum(y_true, dim=axis) + torch.sum(y_predicted, dim=axis) + epsilon
    dice_coefficient = torch.mean((dice_numerator)/(dice_denominator))
    return dice_coefficient

def dsc_loss(y_true, y_predicted, axis=(2,3), epsilon=0.00001):
    return -dice_similarity_coefficient(y_true, y_predicted, axis, epsilon)


class ImageDataset2(Dataset):
    def __init__(self, ip_root, gt_root, ip_size=640):
        self.img_size = (ip_size, ip_size)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.ip_files = sorted(glob.glob(ip_root+"/*.*"))
        self.gt_files = sorted(glob.glob(gt_root + "/*.*"))
        # self.ip_files = ip_root
        # self.gt_files = gt_root

    def __getitem__(self, item):
        ip_img = cv2.imread(self.ip_files[item % len(self.ip_files)]).astype('float32')
        ip_resized = cv2.resize(ip_img, self.img_size, interpolation=cv2.INTER_LINEAR)
        ip_cvt = cv2.cvtColor(ip_resized, cv2.COLOR_BGR2RGB).astype('float32')
        ip = np.moveaxis(ip_cvt, -1, 0)

        gt_img = cv2.imread(self.gt_files[item % len(self.gt_files)], 0)
        gt = gt_filter(gt_img).astype('float32')

        # gt0 = 1 - gt1
        # gt = np.concatenate((gt0, gt1), axis=0)

        # ip_final = self.transform(ip)
        # gt = self.transform(gt)

        # return {"ip": ip_img, "gt": gt_img}
        # ip_tensor = np.concatenate((ip, mask_img), axis=0)
        return ip, gt

    def __len__(self):
        return len(self.ip_files)


root = os.getcwd()

train_img = root + "/datasets/segmentation/trainTRACKimg"
train_gt = root + "/datasets/segmentation/trainTRACKmask"


train_loader = DataLoader(ImageDataset2(train_img, train_gt), batch_size=8, shuffle=True, num_workers=4)
model = UNet(n_channels=3, n_classes=4)

output_dir = root + "/model_weights/"

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

model.to(device)

min_train_loss = np.Inf
epochs_no_improve = 0
n_epochs_stop= 10

training_losses = []

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

for epoch in range(0, 250):
    train_loss = 0.0
    model.train()
    torch.autograd.set_detect_anomaly(True)
    for (input_image, ground_truth) in tqdm(train_loader):
        # images = Variable(data["ip"].type(Tensor))
        # mask = Variable(data["gt"].type(Tensor))
        input_image = input_image.to(device)
        ground_truth = ground_truth.to(device)
        input_image = input_image.float()

        optimizer.zero_grad()
        output = model(input_image)

        # loss = criterion(op_mask, ground_truth)
        loss = dsc_loss(ground_truth, output)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if train_loss < min_train_loss:
        epochs_no_improve = 0
        min_train_loss = train_loss
        torch.save(model.state_dict(), output_dir + "segment_model.pth")
    else:
        epochs_no_improve += 1

    if epochs_no_improve == n_epochs_stop:
        print("early stopping")
        break

    print("[epoch %d] [Training loss: %f] " % (epoch, train_loss/len(train_loader)))
    training_losses.append(train_loss/len(train_loader))

train_numpy = np.array(training_losses)
plt.plot(train_numpy)
plt.savefig(output_dir + 'train_loss.png')
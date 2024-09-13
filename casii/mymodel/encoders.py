import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from TransPath.ctran import ctranspath


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 125 * 125, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # -> n, 3, 32, 32
        #4, 16,125,125 is n batch?
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        #print(x.shape)
        x = x.view(-1, 16 * 125 * 125)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x

def resnet34(outd=1, weights='IMAGENET1K_V1'):
    net = getattr(models, 'resnet34')(weights)
    for name, param in net.named_parameters():
        if name.split('.')[0] not in ['layer4', 'fc']:
            param.requires_grad = False
    net.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(512, outd))
    return net



def resnet50(outd=1, weights=models.ResNet50_Weights.IMAGENET1K_V2):
    net = getattr(models, 'resnet50')(weights)
    for name, param in net.named_parameters():
        if name.split('.')[0] not in ['layer4', 'fc']:
            param.requires_grad = False
    net.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(2048, outd))
    return net


def swin_s(outd=1, weights='IMAGENET1K_V1'):
    net = getattr(models, 'swin_s')(weights=weights)
    for name, param in net.named_parameters():
        if not name.startswith('features.7') or not name.startswith('norm') or not name.startswith('head'):
            param.requires_grad = False
    net.head = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(768, outd))
    return net

def ctrans(outd=1):
    net = ctranspath()
    net.head = nn.Identity()
    td = torch.load(r'./TransPath/ctranspath.pth')
    net.load_state_dict(td['model'], strict=True)
    for name, param in net.named_parameters():
        if not name.startswith('layers.3') or not name.startswith('norm') or not name.startswith('head'):
            param.requires_grad = False
    net.head = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(768, outd))
    return net

def ctpencoder():
    net = ctranspath()
    net.head = nn.Identity()
    td = torch.load(r'./TransPath/ctranspath.pth')
    net.load_state_dict(td['model'], strict=True)

    return net

def vit(outd=1, weights=models.ViT_B_16_Weights.IMAGENET1K_V1):
    net = getattr(models, 'vit_b_16')(weights)
    for name, param in net.named_parameters():
        if name.split('.')[0] not in ['norm_layer']:
            param.requires_grad = False
    net.heads = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(768, outd)) #find right size
    return net

def xception(outd=1, weights=models.ResNet50_Weights.IMAGENET1K_V2):
    net = getattr(models, 'resnet50')(weights)
    for name, param in net.named_parameters():
        if name.split('.')[0] not in ['layer4', 'fc']:
            param.requires_grad = False
    net.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(2048, outd))
    return net

    
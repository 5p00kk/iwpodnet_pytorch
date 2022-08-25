import torch
from torch import nn
from torchsummary import summary

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding='same')
        self.shortcut = nn.Sequential()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=k_size, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = self.bn2(self.conv2(input))
        input = input + shortcut
        return nn.ReLU()(input)

class ConvBatch(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, activation):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding='same')
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, input):
        input = self.bn(self.conv(input))
        if(self.activation == 'relu'):
            input = nn.ReLU()(input)
        return input

class EndBlockIwpod(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.probs_conv_batch1 = ConvBatch(in_channels, 64, 3, 'relu')
        self.probs_conv_batch2 = ConvBatch(64, 32, 3, 'linear')
        self.probs_conv1 = nn.Conv2d(32, 1, 3, padding='same')
        
        self.box_conv_batch1 = ConvBatch(in_channels, 64, 3, 'relu')
        self.box_conv_batch2 = ConvBatch(64, 32, 3, 'linear')
        self.box_conv1 = nn.Conv2d(32, 6, 3, padding='same')

    def forward(self, input):
        probs = self.probs_conv_batch1(input)
        probs = self.probs_conv_batch2(probs)
        probs = nn.Sigmoid()(self.probs_conv1(probs))

        box = self.box_conv_batch1(input)
        box = self.box_conv_batch2(box)
        box = self.box_conv1(box)
        return torch.cat((probs, box), 1)

class iwpod(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer0 = ConvBatch(in_channels, 16, 3, 'relu')
        self.layer1 = ConvBatch(16, 16, 3, 'relu')
        self.layer2 = nn.MaxPool2d(2, 2)
        
        self.layer3 = ConvBatch(16, 32, 3, 'relu')
        self.layer4 = ResBlock(32, 32, 3)
        self.layer5 = nn.MaxPool2d(2, 2)
        
        self.layer6 = ConvBatch(32, 64, 3, 'relu')
        self.layer7 = ResBlock(64, 64, 3)
        self.layer8 = ResBlock(64, 64, 3)
        self.layer9 = nn.MaxPool2d(2, 2)

        self.layer10 = ConvBatch(64, 64, 3, 'relu')
        self.layer11 = ResBlock(64, 64, 3)
        self.layer12 = ResBlock(64, 64, 3)
        self.layer13 = nn.MaxPool2d(2, 2)

        self.layer14 = ConvBatch(64, 128, 3, 'relu')
        self.layer15 = ResBlock(128, 128, 3)
        self.layer16 = ResBlock(128, 128, 3)
        self.layer17 = ResBlock(128, 128, 3)
        self.layer18 = ResBlock(128, 128, 3)

        self.layer19 = EndBlockIwpod(128)
        
    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.layer5(input)
        input = self.layer6(input)
        input = self.layer7(input)
        input = self.layer8(input)
        input = self.layer9(input)
        input = self.layer10(input)
        input = self.layer11(input)
        input = self.layer12(input)
        input = self.layer13(input)
        input = self.layer14(input)
        input = self.layer15(input)
        input = self.layer16(input)
        input = self.layer17(input)
        input = self.layer18(input)
        input = self.layer19(input)
        return input

iwpod_net = iwpod(3)
iwpod_net.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
summary(iwpod_net, (3, 224, 224))


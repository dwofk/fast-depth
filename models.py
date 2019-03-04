import os
import torch
import torch.nn as nn
import torchvision.models
import collections
import math
import torch.nn.functional as F
import imagenet.mobilenet

class Identity(nn.Module):
    # a dummy identity module
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Unpool(nn.Module):
    # Unpool: 2*2 unpooling with zero padding
    def __init__(self, stride=2):
        super(Unpool, self).__init__()

        self.stride = stride

        # create kernel [1, 0; 0, 0]
        self.mask = torch.zeros(1, 1, stride, stride)
        self.mask[:,:,0,0] = 1

    def forward(self, x):
        assert x.dim() == 4
        num_channels = x.size(1)
        return F.conv_transpose2d(x,
            self.mask.detach().type_as(x).expand(num_channels, 1, -1, -1),
            stride=self.stride, groups=num_channels)

def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv(in_channels, out_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=padding,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )

def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )

def convt(in_channels, out_channels, kernel_size):
    stride = 2
    padding = (kernel_size - 1) // 2
    output_padding = kernel_size % 2
    assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,
                stride,padding,output_padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

def convt_dw(channels, kernel_size):
    stride = 2
    padding = (kernel_size - 1) // 2
    output_padding = kernel_size % 2
    assert -2 - 2*padding + kernel_size + output_padding == 0, "deconv parameters incorrect"
    return nn.Sequential(
            nn.ConvTranspose2d(channels,channels,kernel_size,
                stride,padding,output_padding,bias=False,groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

def upconv(in_channels, out_channels):
    return nn.Sequential(
        Unpool(2),
        nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

class upproj(nn.Module):
    # UpProj module has two branches, with a Unpool at the start and a ReLu at the end
    #   upper branch: 5*5 conv -> batchnorm -> ReLU -> 3*3 conv -> batchnorm
    #   bottom branch: 5*5 conv -> batchnorm

    def __init__(self, in_channels, out_channels):
        super(upproj, self).__init__()
        self.unpool = Unpool(2)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=5,stride=1,padding=2,bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.unpool(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return F.relu(x1 + x2)

class Decoder(nn.Module):
    names = ['deconv{}{}'.format(i,dw) for i in range(3,10,2) for dw in ['', 'dw']]
    names.append("upconv")
    names.append("upproj")
    for i in range(3,10,2):
        for dw in ['', 'dw']:
            names.append("nnconv{}{}".format(i, dw))
            names.append("blconv{}{}".format(i, dw))
            names.append("shuffle{}{}".format(i, dw))

class DeConv(nn.Module):

    def __init__(self, kernel_size, dw):
        super(DeConv, self).__init__()
        if dw:
            self.convt1 = nn.Sequential(
                convt_dw(1024, kernel_size),
                pointwise(1024, 512))
            self.convt2 = nn.Sequential(
                convt_dw(512, kernel_size),
                pointwise(512, 256))
            self.convt3 = nn.Sequential(
                convt_dw(256, kernel_size),
                pointwise(256, 128))
            self.convt4 = nn.Sequential(
                convt_dw(128, kernel_size),
                pointwise(128, 64))
            self.convt5 = nn.Sequential(
                convt_dw(64, kernel_size),
                pointwise(64, 32))
        else:
            self.convt1 = convt(1024, 512, kernel_size)
            self.convt2 = convt(512, 256, kernel_size)
            self.convt3 = convt(256, 128, kernel_size)
            self.convt4 = convt(128, 64, kernel_size)
            self.convt5 = convt(64, 32, kernel_size)
        self.convf = pointwise(32, 1)

    def forward(self, x):
        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)
        x = self.convt4(x)
        x = self.convt5(x)
        x = self.convf(x)
        return x


class UpConv(nn.Module):

    def __init__(self):
        super(UpConv, self).__init__()
        self.upconv1 = upconv(1024, 512)
        self.upconv2 = upconv(512, 256)
        self.upconv3 = upconv(256, 128)
        self.upconv4 = upconv(128, 64)
        self.upconv5 = upconv(64, 32)
        self.convf = pointwise(32, 1)

    def forward(self, x):
        x = self.upconv1(x)
        x = self.upconv2(x)
        x = self.upconv3(x)
        x = self.upconv4(x)
        x = self.upconv5(x)
        x = self.convf(x)
        return x

class UpProj(nn.Module):
    # UpProj decoder consists of 4 upproj modules with decreasing number of channels and increasing feature map size

    def __init__(self):
        super(UpProj, self).__init__()
        self.upproj1 = upproj(1024, 512)
        self.upproj2 = upproj(512, 256)
        self.upproj3 = upproj(256, 128)
        self.upproj4 = upproj(128, 64)
        self.upproj5 = upproj(64, 32)
        self.convf = pointwise(32, 1)

    def forward(self, x):
        x = self.upproj1(x)
        x = self.upproj2(x)
        x = self.upproj3(x)
        x = self.upproj4(x)
        x = self.upproj5(x)
        x = self.convf(x)
        return x

class NNConv(nn.Module):

    def __init__(self, kernel_size, dw):
        super(NNConv, self).__init__()
        if dw:
            self.conv1 = nn.Sequential(
                depthwise(1024, kernel_size),
                pointwise(1024, 512))
            self.conv2 = nn.Sequential(
                depthwise(512, kernel_size),
                pointwise(512, 256))
            self.conv3 = nn.Sequential(
                depthwise(256, kernel_size),
                pointwise(256, 128))
            self.conv4 = nn.Sequential(
                depthwise(128, kernel_size),
                pointwise(128, 64))
            self.conv5 = nn.Sequential(
                depthwise(64, kernel_size),
                pointwise(64, 32))
            self.conv6 = pointwise(32, 1)
        else:
            self.conv1 = conv(1024, 512, kernel_size)
            self.conv2 = conv(512, 256, kernel_size)
            self.conv3 = conv(256, 128, kernel_size)
            self.conv4 = conv(128, 64, kernel_size)
            self.conv5 = conv(64, 32, kernel_size)
            self.conv6 = pointwise(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv4(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv5(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv6(x)
        return x

class BLConv(NNConv):

    def __init__(self, kernel_size, dw):
        super(BLConv, self).__init__(kernel_size, dw)

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv5(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = self.conv6(x)
        return x

class ShuffleConv(nn.Module):

    def __init__(self, kernel_size, dw):
        super(ShuffleConv, self).__init__()
        if dw:
            self.conv1 = nn.Sequential(
                depthwise(256, kernel_size),
                pointwise(256, 256))
            self.conv2 = nn.Sequential(
                depthwise(64, kernel_size),
                pointwise(64, 64))
            self.conv3 = nn.Sequential(
                depthwise(16, kernel_size),
                pointwise(16, 16))
            self.conv4 = nn.Sequential(
                depthwise(4, kernel_size),
                pointwise(4, 4))
        else:
            self.conv1 = conv(256, 256, kernel_size)
            self.conv2 = conv(64, 64, kernel_size)
            self.conv3 = conv(16, 16, kernel_size)
            self.conv4 = conv(4, 4, kernel_size)

    def forward(self, x):
        x = F.pixel_shuffle(x, 2)
        x = self.conv1(x)

        x = F.pixel_shuffle(x, 2)
        x = self.conv2(x)

        x = F.pixel_shuffle(x, 2)
        x = self.conv3(x)

        x = F.pixel_shuffle(x, 2)
        x = self.conv4(x)

        x = F.pixel_shuffle(x, 2)
        return x

def choose_decoder(decoder):
    depthwise = ('dw' in decoder)
    if decoder[:6] == 'deconv':
        assert len(decoder)==7 or (len(decoder)==9 and 'dw' in decoder)
        kernel_size = int(decoder[6])
        model = DeConv(kernel_size, depthwise)
    elif decoder == "upproj":
        model = UpProj()
    elif decoder == "upconv":
        model = UpConv()
    elif decoder[:7] == 'shuffle':
        assert len(decoder)==8 or (len(decoder)==10 and 'dw' in decoder)
        kernel_size = int(decoder[7])
        model = ShuffleConv(kernel_size, depthwise)
    elif decoder[:6] == 'nnconv':
        assert len(decoder)==7 or (len(decoder)==9 and 'dw' in decoder)
        kernel_size = int(decoder[6])
        model = NNConv(kernel_size, depthwise)
    elif decoder[:6] == 'blconv':
        assert len(decoder)==7 or (len(decoder)==9 and 'dw' in decoder)
        kernel_size = int(decoder[6])
        model = BLConv(kernel_size, depthwise)
    else:
        assert False, "invalid option for decoder: {}".format(decoder)
    model.apply(weights_init)
    return model


class ResNet(nn.Module):
    def __init__(self, layers, decoder, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))
        
        super(ResNet, self).__init__()
        self.output_size = output_size
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)
        if not pretrained:
            pretrained_model.apply(weights_init)
        
        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)
        
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048
        self.conv2 = nn.Conv2d(num_channels, 1024, 1)
        weights_init(self.conv2)
        self.decoder = choose_decoder(decoder)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv2(x)

        # decoder
        x = self.decoder(x)

        return x

class MobileNet(nn.Module):
    def __init__(self, decoder, output_size, in_channels=3, pretrained=True):

        super(MobileNet, self).__init__()
        self.output_size = output_size
        mobilenet = imagenet.mobilenet.MobileNet()
        if pretrained:
            pretrained_path = os.path.join('imagenet', 'results', 'imagenet.arch=mobilenet.lr=0.1.bs=256', 'model_best.pth.tar')
            checkpoint = torch.load(pretrained_path)
            state_dict = checkpoint['state_dict']

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            mobilenet.load_state_dict(new_state_dict)
        else:
            mobilenet.apply(weights_init)

        if in_channels == 3:
            self.mobilenet = nn.Sequential(*(mobilenet.model[i] for i in range(14)))
        else:
            def conv_bn(inp, oup, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU6(inplace=True)
                )

            self.mobilenet = nn.Sequential(
                conv_bn(in_channels,  32, 2),
                *(mobilenet.model[i] for i in range(1,14))
                )

        self.decoder = choose_decoder(decoder)

    def forward(self, x):
        x = self.mobilenet(x)
        x = self.decoder(x)
        return x

class ResNetSkipAdd(nn.Module):
    def __init__(self, layers, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))
        
        super(ResNetSkipAdd, self).__init__()
        self.output_size = output_size
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)
        if not pretrained:
            pretrained_model.apply(weights_init)
        
        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)
        
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048
        self.conv2 = nn.Conv2d(num_channels, 1024, 1)
        weights_init(self.conv2)
        
        kernel_size = 5
        self.decode_conv1 = conv(1024, 512, kernel_size)
        self.decode_conv2 = conv(512, 256, kernel_size)
        self.decode_conv3 = conv(256, 128, kernel_size)
        self.decode_conv4 = conv(128, 64, kernel_size)
        self.decode_conv5 = conv(64, 32, kernel_size)
        self.decode_conv6 = pointwise(32, 1)
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        # print("x1", x1.size())
        x2 = self.maxpool(x1)
        # print("x2", x2.size())
        x3 = self.layer1(x2)
        # print("x3", x3.size())
        x4 = self.layer2(x3)
        # print("x4", x4.size())
        x5 = self.layer3(x4)
        # print("x5", x5.size())
        x6 = self.layer4(x5)
        # print("x6", x6.size())
        x7 = self.conv2(x6)

        # decoder
        y10 = self.decode_conv1(x7)
        # print("y10", y10.size())
        y9 = F.interpolate(y10 + x6, scale_factor=2, mode='nearest')
        # print("y9", y9.size())
        y8 = self.decode_conv2(y9)
        # print("y8", y8.size())
        y7 = F.interpolate(y8 + x5, scale_factor=2, mode='nearest')
        # print("y7", y7.size())
        y6 = self.decode_conv3(y7)
        # print("y6", y6.size())
        y5 = F.interpolate(y6 + x4, scale_factor=2, mode='nearest')
        # print("y5", y5.size())
        y4 = self.decode_conv4(y5)
        # print("y4", y4.size())
        y3 = F.interpolate(y4 + x3, scale_factor=2, mode='nearest')
        # print("y3", y3.size())
        y2 = self.decode_conv5(y3 + x1)
        # print("y2", y2.size())
        y1 = F.interpolate(y2, scale_factor=2, mode='nearest')
        # print("y1", y1.size())
        y = self.decode_conv6(y1)

        return y

class ResNetSkipConcat(nn.Module):
    def __init__(self, layers, output_size, in_channels=3, pretrained=True):

        if layers not in [18, 34, 50, 101, 152]:
            raise RuntimeError('Only 18, 34, 50, 101, and 152 layer model are defined for ResNet. Got {}'.format(layers))
        
        super(ResNetSkipConcat, self).__init__()
        self.output_size = output_size
        pretrained_model = torchvision.models.__dict__['resnet{}'.format(layers)](pretrained=pretrained)
        if not pretrained:
            pretrained_model.apply(weights_init)
        
        if in_channels == 3:
            self.conv1 = pretrained_model._modules['conv1']
            self.bn1 = pretrained_model._modules['bn1']
        else:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            weights_init(self.conv1)
            weights_init(self.bn1)
        
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']
        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        # clear memory
        del pretrained_model

        # define number of intermediate channels
        if layers <= 34:
            num_channels = 512
        elif layers >= 50:
            num_channels = 2048
        self.conv2 = nn.Conv2d(num_channels, 1024, 1)
        weights_init(self.conv2)
        
        kernel_size = 5
        self.decode_conv1 = conv(1024, 512, kernel_size)
        self.decode_conv2 = conv(768, 256, kernel_size)
        self.decode_conv3 = conv(384, 128, kernel_size)
        self.decode_conv4 = conv(192, 64, kernel_size)
        self.decode_conv5 = conv(128, 32, kernel_size)
        self.decode_conv6 = pointwise(32, 1)
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, x):
        # resnet
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        # print("x1", x1.size())
        x2 = self.maxpool(x1)
        # print("x2", x2.size())
        x3 = self.layer1(x2)
        # print("x3", x3.size())
        x4 = self.layer2(x3)
        # print("x4", x4.size())
        x5 = self.layer3(x4)
        # print("x5", x5.size())
        x6 = self.layer4(x5)
        # print("x6", x6.size())
        x7 = self.conv2(x6)

        # decoder
        y10 = self.decode_conv1(x7)
        # print("y10", y10.size())
        y9 = F.interpolate(y10, scale_factor=2, mode='nearest')
        # print("y9", y9.size())
        y8 = self.decode_conv2(torch.cat((y9, x5), 1))
        # print("y8", y8.size())
        y7 = F.interpolate(y8, scale_factor=2, mode='nearest')
        # print("y7", y7.size())
        y6 = self.decode_conv3(torch.cat((y7, x4), 1))
        # print("y6", y6.size())
        y5 = F.interpolate(y6, scale_factor=2, mode='nearest')
        # print("y5", y5.size())
        y4 = self.decode_conv4(torch.cat((y5, x3), 1))
        # print("y4", y4.size())
        y3 = F.interpolate(y4, scale_factor=2, mode='nearest')
        # print("y3", y3.size())
        y2 = self.decode_conv5(torch.cat((y3, x1), 1))
        # print("y2", y2.size())
        y1 = F.interpolate(y2, scale_factor=2, mode='nearest')
        # print("y1", y1.size())
        y = self.decode_conv6(y1)

        return y

class MobileNetSkipAdd(nn.Module):
    def __init__(self, output_size, pretrained=True):

        super(MobileNetSkipAdd, self).__init__()
        self.output_size = output_size
        mobilenet = imagenet.mobilenet.MobileNet()
        if pretrained:
            pretrained_path = os.path.join('imagenet', 'results', 'imagenet.arch=mobilenet.lr=0.1.bs=256', 'model_best.pth.tar')
            checkpoint = torch.load(pretrained_path)
            state_dict = checkpoint['state_dict']

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            mobilenet.load_state_dict(new_state_dict)
        else:
            mobilenet.apply(weights_init)

        for i in range(14):
            setattr( self, 'conv{}'.format(i), mobilenet.model[i])

        kernel_size = 5
        # self.decode_conv1 = conv(1024, 512, kernel_size)
        # self.decode_conv2 = conv(512, 256, kernel_size)
        # self.decode_conv3 = conv(256, 128, kernel_size)
        # self.decode_conv4 = conv(128, 64, kernel_size)
        # self.decode_conv5 = conv(64, 32, kernel_size)
        self.decode_conv1 = nn.Sequential(
            depthwise(1024, kernel_size),
            pointwise(1024, 512))
        self.decode_conv2 = nn.Sequential(
            depthwise(512, kernel_size),
            pointwise(512, 256))
        self.decode_conv3 = nn.Sequential(
            depthwise(256, kernel_size),
            pointwise(256, 128))
        self.decode_conv4 = nn.Sequential(
            depthwise(128, kernel_size),
            pointwise(128, 64))
        self.decode_conv5 = nn.Sequential(
            depthwise(64, kernel_size),
            pointwise(64, 32))
        self.decode_conv6 = pointwise(32, 1)
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, x):
        # skip connections: dec4: enc1
        # dec 3: enc2 or enc3
        # dec 2: enc4 or enc5
        for i in range(14):
            layer = getattr(self, 'conv{}'.format(i))
            x = layer(x)
            # print("{}: {}".format(i, x.size()))
            if i==1:
                x1 = x
            elif i==3:
                x2 = x
            elif i==5:
                x3 = x
        for i in range(1,6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i==4:
                x = x + x1
            elif i==3:
                x = x + x2
            elif i==2:
                x = x + x3
            # print("{}: {}".format(i, x.size()))
        x = self.decode_conv6(x)
        return x

class MobileNetSkipConcat(nn.Module):
    def __init__(self, output_size, pretrained=True):

        super(MobileNetSkipConcat, self).__init__()
        self.output_size = output_size
        mobilenet = imagenet.mobilenet.MobileNet()
        if pretrained:
            pretrained_path = os.path.join('imagenet', 'results', 'imagenet.arch=mobilenet.lr=0.1.bs=256', 'model_best.pth.tar')
            checkpoint = torch.load(pretrained_path)
            state_dict = checkpoint['state_dict']

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            mobilenet.load_state_dict(new_state_dict)
        else:
            mobilenet.apply(weights_init)

        for i in range(14):
            setattr( self, 'conv{}'.format(i), mobilenet.model[i])

        kernel_size = 5
        # self.decode_conv1 = conv(1024, 512, kernel_size)
        # self.decode_conv2 = conv(512, 256, kernel_size)
        # self.decode_conv3 = conv(256, 128, kernel_size)
        # self.decode_conv4 = conv(128, 64, kernel_size)
        # self.decode_conv5 = conv(64, 32, kernel_size)
        self.decode_conv1 = nn.Sequential(
            depthwise(1024, kernel_size),
            pointwise(1024, 512))
        self.decode_conv2 = nn.Sequential(
            depthwise(512, kernel_size),
            pointwise(512, 256))
        self.decode_conv3 = nn.Sequential(
            depthwise(512, kernel_size),
            pointwise(512, 128))
        self.decode_conv4 = nn.Sequential(
            depthwise(256, kernel_size),
            pointwise(256, 64))
        self.decode_conv5 = nn.Sequential(
            depthwise(128, kernel_size),
            pointwise(128, 32))
        self.decode_conv6 = pointwise(32, 1)
        weights_init(self.decode_conv1)
        weights_init(self.decode_conv2)
        weights_init(self.decode_conv3)
        weights_init(self.decode_conv4)
        weights_init(self.decode_conv5)
        weights_init(self.decode_conv6)

    def forward(self, x):
        # skip connections: dec4: enc1
        # dec 3: enc2 or enc3
        # dec 2: enc4 or enc5
        for i in range(14):
            layer = getattr(self, 'conv{}'.format(i))
            x = layer(x)
            # print("{}: {}".format(i, x.size()))
            if i==1:
                x1 = x
            elif i==3:
                x2 = x
            elif i==5:
                x3 = x
        for i in range(1,6):
            layer = getattr(self, 'decode_conv{}'.format(i))
            # print("{}a: {}".format(i, x.size()))
            x = layer(x)
            # print("{}b: {}".format(i, x.size()))
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i==4:
                x = torch.cat((x, x1), 1)
            elif i==3:
                x = torch.cat((x, x2), 1)
            elif i==2:
                x = torch.cat((x, x3), 1)
            # print("{}c: {}".format(i, x.size()))
        x = self.decode_conv6(x)
        return x

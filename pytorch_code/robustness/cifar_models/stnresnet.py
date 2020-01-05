'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class SequentialWithArgs(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l-1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, fake_relu=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if fake_relu:
            return FakeReLU.apply(out)
        return F.relu(out)


class ResNet(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    def __init__(self, block, num_blocks, num_classes=10, feat_scale=1, wm=1):
        super(ResNet, self).__init__()

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(feat_scale*widths[3]*block.expansion, num_classes)

        ##############  Spatial Transformer Networks specific   ##############
        # Spatial transformer localization-network
        self._ksize = 3
        self.stn_localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False),  # size : [1x3x32x32]
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )

        # Regressor for the 3 * 2 affine matrix
        self.stn_fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 6)
            # nn.Linear(10 * 3 * 3, 32),
            # nn.ReLU(True),
            # nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.stn_fc_loc[2].weight.data.zero_()
        self.stn_fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)

    def stn(self, x):
        xs = self.stn_localization(x)
        # print("Pre view size:{}".format(xs.size()))

        xs = xs.view(-1, 32*4*4)
        theta = self.stn_fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert (not no_relu),  \
            "no_relu not yet supported for this architecture"

        # transform the input via STN module
        x = self.stn(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out, fake_relu=fake_relu)
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        if with_latent:
            return final, pre_out
        return final

def stnResNet18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], **kwargs)

def stnResNet18Wide(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], wd=1.5, **kwargs)

def stnResNet18Thin(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2], wd=.75, **kwargs)

def stnResNet34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3], **kwargs)

def stnResNet50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3], **kwargs)

def stnResNet101(**kwargs):
    return ResNet(Bottleneck, [3,4,23,3], **kwargs)

def stnResNet152(**kwargs):
    return ResNet(Bottleneck, [3,8,36,3], **kwargs)

stnresnet50 = stnResNet50
stnresnet34 = stnResNet34
stnresnet18 = stnResNet18
stnresnet101 = stnResNet101
stnresnet152 = stnResNet152

# resnet18thin = ResNet18Thin
# resnet18wide = ResNet18Wide
def test():
    net = stnResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())


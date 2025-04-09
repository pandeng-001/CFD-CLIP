# Network Structure

import torch.nn as nn
import torch.nn.functional  as F
import torch


# define ResBlock
class ResBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super(ResBlock, self).__init__()
        # Contains two consecutive convolutional layers
        self.block_conv=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(outchannel,outchannel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(outchannel)
        )
 
        # shortcut 
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                # Uses 1x1 conv for dimension adjustment
                # Skip connections activate when stride!=1
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
 
    def forward(self,x):
        out1=self.block_conv(x)
        out2=self.shortcut(x)+out1
        out2=F.relu(out2) 
        return out2
 
 
# define RESNET18
class ResNet_18(nn.Module):
    def __init__(self,ResBlock,num_classes):
        super(ResNet_18, self).__init__()
 
        self.in_channels = 64 # layer1  channel
        # Initial standalone convolutional layer
        self.conv1=nn.Sequential(
            # (n-f+2*p)/s+1,n=28,n=32
            # nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), #64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0) #64
            # nn.Dropout(0.25)
        )
 
        self.layer1=self.make_layer(ResBlock,64,2,stride=1) #64
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2) #32
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2) #16
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2) #8

        # torch.nn.AdaptiveAvgPool2d() takes two parameters: 
        # output_height and output_width
        # - Preserves original channel count
        # - Here forces spatial dimensions to 1×1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

        # self.linear=nn.Linear(2*2*512,512)
        # self.linear2=nn.Linear(512,100)
 
        self.linear=nn.Linear(512*1*1,num_classes)
 
        self.dropout = nn.Dropout(0.3)
 
    # repeatedly stack identical residual blocks
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x=self.conv1(x)
        # x=self.dropout(x)
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x=self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=self.linear(x)
        x=self.dropout(x)
 
        return x


# define RESNET34
class ResNet_34(nn.Module):
    def __init__(self,ResBlock,num_classes):
        super(ResNet_34, self).__init__()
 
        self.in_channels = 64 # input layer1  channel
        # Initial standalone convolutional layer
        self.conv1=nn.Sequential(
            # (n-f+2*p)/s+1,n=28,n=32
            # nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), #64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0) #64
            # nn.Dropout(0.25)
        )
 
        self.layer1=self.make_layer(ResBlock, 64, 4, stride=1) #64
        self.layer2 = self.make_layer(ResBlock, 128, 4, stride=2) #32
        self.layer3 = self.make_layer(ResBlock, 256, 4, stride=2) #16
        self.layer4 = self.make_layer(ResBlock, 512, 4, stride=2) #8
 
        # torch.nn.AdaptiveAvgPool2d() takes two parameters: 
        # output_height and output_width
        # - Preserves original channel count
        # - Here forces spatial dimensions to 1×1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 
        
        # self.linear=nn.Linear(2*2*512,512)
        # self.linear2=nn.Linear(512,100)
 
        self.linear=nn.Linear(512*1*1,num_classes)
 
        self.dropout = nn.Dropout(0.3)
 
    # repeatedly stack identical residual blocks
    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x=self.conv1(x)
        # x=self.dropout(x)
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x=self.avgpool(x)
        x = x.view(x.size(0), -1)
        x=self.linear(x)
        x=self.dropout(x)
 
        return x

# WRN BasicBlock
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# define WRN model
class WideResNet(nn.Module):
    def __init__(self, depth, width_factor, num_classes=100):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4"
        n = (depth - 4) // 6
        k = width_factor
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(16 * k, n, stride=1)
        self.layer2 = self._make_layer(32 * k, n, stride=2)
        self.layer3 = self._make_layer(64 * k, n, stride=2)
        self.bn = nn.BatchNorm2d(64 * k)
        self.fc = nn.Linear(64 * k, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def _make_layer(self, out_channels, blocks, stride):
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fc(out)

# Build activation function: HSwish
# Similar to x*sigmoid() but with modified formulation
class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

# Squeeze-and-Excitation Module  
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)

    def forward(self, x):
        scale = x.mean((2, 3), keepdim=True)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale

# MBConv block with expand ratio
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, se=False, activation=HSwish):
        super(MBConv, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.se = SqueezeExcitation(hidden_dim) if se else nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            activation(),
            self.se,
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# MobileNetV3 Dynamic Model
class MobileNetV3Dynamic(nn.Module):
    def __init__(self, num_classes=100, width_mult=1.0, depth_mult=1.0, input_size=32):
        super(MobileNetV3Dynamic, self).__init__()
        self.width_mult = width_mult
        self.depth_mult = depth_mult

        def make_divisible(v, divisor=8, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        # Stage configurations: (expand_ratio, out_channels, num_blocks, stride, use_se)
        cfgs = [
            # expand_ratio, out_channels, num_blocks, stride, use_se
            (1, 16, 1, 1, False),
            (4, 24, 2, 2, False),
            (3, 40, 3, 2, True),
            (6, 80, 4, 2, False),
            (6, 112, 3, 1, True),
            (6, 160, 3, 2, True),
        ]

        input_channel = make_divisible(16 * width_mult)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2 if input_size > 32 else 1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            HSwish(),
        )

        # Build MBConv blocks
        self.blocks = []
        for t, c, n, s, se in cfgs:
            output_channel = make_divisible(c * width_mult)
            num_blocks = max(1, int(n * depth_mult))
            for i in range(num_blocks):
                stride = s if i == 0 else 1
                self.blocks.append(MBConv(input_channel, output_channel, stride, expand_ratio=t, se=se))
                input_channel = output_channel
        self.blocks = nn.Sequential(*self.blocks)

        # Final layers
        last_channel = make_divisible(960 * width_mult)
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channel),
            HSwish(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(last_channel, 1280),
            HSwish(),
            nn.Linear(1280, num_classes),
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.dropout(x)

# Function to initialize the model with dynamic parameters
def get_mobilenetv3(num_classes, width_mult=1.0, depth_mult=1.0, input_size=32):
    return MobileNetV3Dynamic(num_classes=num_classes, width_mult=width_mult, depth_mult=depth_mult, input_size=input_size)


if __name__ == "__main__":
    # test resnet model
    inputs = torch.randn([8, 3, 32, 32])
    resnet34 = ResNet_34(ResBlock, num_classes=100)
    print(resnet34(inputs))
    # Instantiate WRN-28-10
    model = WideResNet(depth=16, width_factor=8, num_classes=100)
    print(model(inputs))
    # test mobilenet model 
    mobilenetV3 = get_mobilenetv3(100)
    print(mobilenetV3(inputs))

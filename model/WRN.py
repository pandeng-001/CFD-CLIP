import torch.nn as nn
import torch.nn.functional  as F
import torch


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
        # feature = out
        out = self.layer2(out)
        feature = out
        out = self.layer3(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return feature, self.fc(out)


if __name__ == "__main__":
    inputa = torch.randn([1, 3, 32, 32]) 
    model= WideResNet(depth=16, width_factor=8)
    # model= WideResNet(depth=28, width_factor=10)
    f, output = model(inputa)
    print(f.shape, output.shape)

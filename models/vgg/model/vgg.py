import torch.nn as nn

cfg = {
    "VGG11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "VGG13": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    "VGG16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    "VGG19": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class ConvBlock(nn.Module):  # encapsulate convolution layer followed by BN and ReLU

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class FullBlock(nn.Module):  # encapsulate convolution layer followed by ReLU

    def __init__(self, input_dim, output_dim):
        super(FullBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class VGG(nn.Module):  # general VGG model

    def __init__(self, in_channels, class_num, version):
        """

        :param in_channels: initial image channels
        :param class_num:   number of classifications
        :param version:     vgg version
        """
        super(VGG, self).__init__()
        assert version in cfg
        # if version not in cfg:
        #     raise ValueError("wrong model version")
        self.sequence = cfg[version]
        self.class_num = class_num
        self.features = self._generate_conv(in_channels)
        self.classifier = self._generate_full()

    def _generate_conv(self, in_channels):
        layers = []
        for x in self.sequence:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(ConvBlock(in_channels, x, 3, 1))
                in_channels = x

        return nn.Sequential(*layers)

    def _generate_full(self):
        """
        Since the images in CIFAR10 training dataset having sizes of 32 x 32,
        after passing through five pooling layers(kernel is 2 x 2 and stride is 2),
        the sizes of output images inevitably are 1 x 1.
        Therefore,
        after passing through all convolutional layers,
        the dimension of output tensor is like N x 512 x 1 x 1 and 512 is number of channels.
        Finally,
        we need to modify dimension of tensor from N x 512 x 1 x 1 to N x 512 and feed tensor into connection layers.
        """

        layers = [FullBlock(512, 4096)]
        nn.Dropout()
        layers.append(FullBlock(4096, 4096))
        nn.Dropout()
        layers.append(FullBlock(4096, self.class_num))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)

    def get(self):
        print("3333")


__all__ = ["VGG"]

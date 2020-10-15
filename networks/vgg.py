import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self, input_dim=1, num_classes=2):
        super(VGG16, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        output = self.conv1(input)
        output = F.relu(output)

        output = self.conv2(output)
        output = F.relu(output)
        output = self.pool(output)

        output = self.conv3(output)
        output = F.relu(output)
        output = self.conv4(output)
        output = F.relu(output)
        output = self.pool(output)

        output = self.conv5(output)
        output = F.relu(output)

        output = self.conv6(output)
        output = F.relu(output)

        output = self.conv6(output)
        output = F.relu(output)
        output = self.pool(output)

        output = self.conv7(output)
        output = F.relu(output)

        output = self.conv8(output)
        output = F.relu(output)

        output = self.conv8(output)
        output = F.relu(output)
        output = self.pool(output)

        output = self.conv8(output)
        output = F.relu(output)

        output = self.conv8(output)
        output = F.relu(output)

        output = self.conv8(output)
        output = F.relu(output)
        output = self.pool(output)

        output = output.view(output.size(0), -1)
        output = self.classifier(output)
        return output

    def get_input_dim(self):
        return self.input_dim

    def get_num_classes(self):
        return self.num_classes

# def make_layers(cfg, batch_norm=False):
#     layers = []
#     in_channels = 3
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     return nn.Sequential(*layers)
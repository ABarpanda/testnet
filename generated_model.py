import torch
import torch.nn as nn

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.5),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = CustomNet()
    print(model)

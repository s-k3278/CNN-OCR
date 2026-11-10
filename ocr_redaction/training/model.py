"""Compact CRNN/CTC OCR model optimized for CPU deployment."""
import torch
from torch import nn


class CRNNOCR(nn.Module):
    def __init__(self, num_classes, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),              # H/2, W/2

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),              # H/4, W/4

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),    # H/8, W/4

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((4, 1), (4, 1)),    # H/32, W/4
        )
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden,
            num_layers=2,
            bidirectional=True,
            dropout=0.15,
        )
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)                      # B,C,1,T
        x = x.squeeze(2).permute(2, 0, 1)    # T,B,C
        x, _ = self.rnn(x)
        return self.classifier(x)

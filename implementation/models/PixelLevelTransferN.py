import torch 
import torch.nn as nn

class PixelLevelTransferN(nn.Module):
    def __init__(self, intensity, in_channels=1, out_channels=1):
        super(PixelLevelTransferN, self).__init__()

        self.intensity = intensity

        ## encoder:
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2, stride=2),
            nn.LeakyReLU())
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))
        ## decoder:
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, out_channels, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.Tanh())
    
    def forward(self, x):
        out = self.enc1(x)
        out = self.enc2(out)
        out = self.enc3(out)
        out = self.enc4(out)
        out = self.dec1(out)
        out = self.dec2(out)
        out = self.dec3(out)
        out = self.dec4(out)
        # out = out.clamp(min=-self.intensity, max=self.intensity)
        out = out.clamp(min=-0.2, max=0.2)
        out = out * (self.intensity / 0.2)
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.regnet.modules import XBlock as ConvBlock

class UNetDecoder(nn.Module):
    """
    UNet style Decoder, suitable for segmentation tasks.

    Parameters:
    -----------    
    config: Configuration dict containing num_features, in_channels, etc. (check config.yaml)
    input_features (list of torch.Tensor): Fused features at various stages.

    Returns:
    --------
    outputs (torch.Tensor): Tensor of shape (out_channels, H, W) containing segmentation maps.
    """
    def __init__(self, config):
        super().__init__()        
        num_features = config["NUM_FEATURES"]
        in_channels = config["IN_CHANNELS"]
        out_channels = config["OUT_CHANNELS"]
        self.H = config["HEIGHT"]
        self.W = config["WIDTH"]
        self.conv_blocks = nn.ModuleList()
        
        C = in_channels
        out_C = C // 2
        self.conv_blocks.append(ConvBlock(C, out_C))
        
        for i in range(1, num_features):
            self.conv_blocks.append(ConvBlock(C + out_C, out_C))

        self.conv1 = ConvBlock(C + out_C, 32)
        self.conv2 = ConvBlock(32, 16)
        self.conv4 = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, encoder_features):
        f = encoder_features[0]
        for i in range(len(encoder_features[:-1])):
            f = self.conv_blocks[i](f)
            _, _, H, W = encoder_features[i+1].shape
            f = F.interpolate(f, size=(H, W), mode="bilinear", align_corners=True)
            f = torch.cat([encoder_features[i+1], f], dim=1)
        out = self.conv1(f)
        out = self.conv2(out)
        out = F.interpolate(out, size=(self.H, self.W), mode="bilinear", align_corners=True)
        out = self.conv4(out)
        return out
    
class ChannelEqualizer(nn.Module):
    """Make all channels equal to out_channels"""
    def __init__(self, in_channels_list, out_channels=32):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        for ch in in_channels_list:
            self.conv_blocks.append(nn.Conv2d(ch, out_channels, kernel_size=3, padding=1))

    def forward(self, features):
        out_features = []
        for i, f in enumerate(features):
            out = self.conv_blocks[i](f)
            out_features.append(out)
        return out_features

class SegmentationHead(nn.Module):
    """
    Segmentation head for lane and drivable area detection.

    Parameters:
    -----------    
    config: Configuration dict containing num_features, in_channels, etc. (check config.yaml)
    fused_features (list of torch.Tensor): Fused features from the BiFPN.
    stem_features (list of torch.Tensor): Stem features from the RegNet. (for spatial information)

    Returns:
    --------
    outputs (torch.Tensor): Tensor of shape (out_channels, H, W) containing segmentation maps.
    """
    def __init__(self, config):
        super().__init__()
        self.equalizer = ChannelEqualizer(config["STEM_IN_CHANNELS_LIST"], config["IN_CHANNELS"])
        self.decoder = UNetDecoder(config)
        self.loss = nn.BCELoss()

    def forward(self, fusedFeatures, stemFeatures):
        stemFeatures = self.equalizer(stemFeatures)
        encoder_features = stemFeatures + fusedFeatures
        features = encoder_features[::-1]
        outputs = self.decoder(features)
        return outputs
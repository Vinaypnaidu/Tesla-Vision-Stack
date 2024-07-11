import torch.nn as nn
from .modules import BiFPNLayer

class BiFPN(nn.Module):
    """
    A multi-layer BiFPN as explained in the EfficientDet (Tan et al. 2020) paper.

    Parameters:
    -----------    
    config: Configuration dict containing num_layers, num_channels, etc. (check config.yaml)
    input_features (list of torch.Tensor): Regnet features at various stages.

    Returns:
    --------
    list of torch.Tensor: Fused features.
    """
    def __init__(self, config):
        super(BiFPN, self).__init__()
        num_layers = config["NUM_LAYERS_BIFPN"]
        num_channels = config["NUM_CHANNELS_BIFPN"]
        in_channels = config["IN_CHANNELS"]
        self.conv1 = nn.Conv2d(in_channels[0], num_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels[1], num_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels[2], num_channels, kernel_size=1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels[3], num_channels, kernel_size=1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(in_channels[4], num_channels, kernel_size=1, stride=1, padding=0)
        self.layers = nn.ModuleList([BiFPNLayer(num_channels) for _ in range(num_layers)])

    def forward(self, input_features):
        features1, features2, features3, features4, features5 = input_features
        features1 = self.conv1(features1)
        features2 = self.conv2(features2)
        features3 = self.conv3(features3)
        features4 = self.conv4(features4)
        features5 = self.conv5(features5)
        features = [features1, features2, features3, features4, features5]
        for layer in self.layers:
            features = layer(features)
        return features
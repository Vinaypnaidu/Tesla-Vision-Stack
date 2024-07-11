import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Conv-BN-Relu. 
    """
    def __init__(self, num_channels, num_groups=None):
        super(ConvBlock, self).__init__()
        if num_groups:
            assert num_channels % num_groups == 0, "num_channels should be divisible by num_groups"
            self.conv = nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, groups=num_groups),
                nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(num_features=num_channels),
                nn.ReLU()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=num_channels),
                nn.ReLU()
            )

    def forward(self, input):
        return self.conv(input)
    

class BiFPNNode(nn.Module):
    """
    A single BiFPN Node.

    Parameters:
    -----------    
    num_channels (int): Number of output channels.
    num_inputs (int): Number of inputs (2 or 3).

    Returns:
    --------
    torch.Tensor: Output features after weighted combination and convolution.
    """
    def __init__(self, num_channels, num_inputs, epsilon=1e-4, num_groups=None):
        super(BiFPNNode, self).__init__()
        self.num_inputs = num_inputs
        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(num_inputs), requires_grad=True)
        self.relu = nn.ReLU()
        self.conv_block = ConvBlock(num_channels, num_groups=num_groups)

    def forward(self, *inputs):
        weights = self.relu(self.weights)
        weights = weights / (weights.sum() + self.epsilon)
        weighted_sum = torch.zeros_like(inputs[0])
        for input, weight in zip(inputs, weights):
            weighted_sum += weight * input
        return self.conv_block(weighted_sum)
    

class BiFPNLayer(nn.Module):
    """
    A BiFPN layer as explained in the EfficientDet (Tan et al. 2020) paper.

    Parameters:
    -----------    
    num_channels (int): Number of output channels.
    input_features (list of torch.Tensor): Regnet features at various stages.

    Returns:
    --------
    list of torch.Tensor: Fused features.
    """
    def __init__(self, num_channels):
        super(BiFPNLayer, self).__init__()
        self.stage4_td_node = BiFPNNode(num_channels=num_channels, num_inputs=2)
        self.stage3_td_node = BiFPNNode(num_channels=num_channels, num_inputs=2)
        self.stage2_td_node = BiFPNNode(num_channels=num_channels, num_inputs=2)

        self.stage1_out_node = BiFPNNode(num_channels=num_channels, num_inputs=2)
        self.stage2_out_node = BiFPNNode(num_channels=num_channels, num_inputs=3)
        self.stage3_out_node = BiFPNNode(num_channels=num_channels, num_inputs=3)
        self.stage4_out_node = BiFPNNode(num_channels=num_channels, num_inputs=3)
        self.stage5_out_node = BiFPNNode(num_channels=num_channels, num_inputs=2)

    def forward(self, input_features):
        self.input_features = input_features
        stage1_in, stage2_in, stage3_in, stage4_in, stage5_in = self.input_features
        stage4_td = self.stage4_td_node(stage4_in, F.interpolate(stage5_in, size=stage4_in.shape[2:], mode='nearest', align_corners=None))
        stage3_td = self.stage3_td_node(stage3_in, F.interpolate(stage4_td, size=stage3_in.shape[2:], mode='nearest', align_corners=None))
        stage2_td = self.stage2_td_node(stage2_in, F.interpolate(stage3_td, size=stage2_in.shape[2:], mode='nearest', align_corners=None))

        stage1_out = self.stage1_out_node(stage1_in, F.interpolate(stage2_td, size=stage1_in.shape[2:], mode='nearest', align_corners=None))
        stage2_out = self.stage2_out_node(stage2_in, stage2_td, F.interpolate(stage1_out, size=stage2_in.shape[2:], mode='nearest', align_corners=None))
        stage3_out = self.stage3_out_node(stage3_in, stage3_td, F.interpolate(stage2_out, size=stage3_in.shape[2:], mode='nearest', align_corners=None))
        stage4_out = self.stage4_out_node(stage4_in, stage4_td, F.interpolate(stage3_out, size=stage4_in.shape[2:], mode='nearest', align_corners=None))
        stage5_out = self.stage5_out_node(stage5_in, F.interpolate(stage4_out, size=stage5_in.shape[2:], mode='nearest', align_corners=None))
        return [stage1_out, stage2_out, stage3_out, stage4_out, stage5_out]
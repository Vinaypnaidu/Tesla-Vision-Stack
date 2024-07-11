import torch
import torch.nn as nn
from src.regnet.regnet import RegNet
from src.bifpn.bifpn import BiFPN
from src.heads.detection.model import ObjectDetectionHead
from src.heads.segmentation.model import SegmentationHead
from src.heads.detection.loss import FocalLoss
from src.heads.segmentation.loss import FocalLossSeg

class HydraNet(nn.Module):
    """
    HydraNet model as demonstrated on Tesla AI Day, 2021. Features a 
    RegNet (Radosavovic et al. 2020) backbone, a BiFPN (Tan et al. 2020)
    for multi-scale feature fusion and task specific heads for object 
    detection, lane detection, drivable area segmentation etc. 

    Parameters:
    -----------    
    config: configuration loaded from yaml file.

    images (torch.Tensor): A batch of images.
    det_annotations (torch.Tensor): Ground truth boxes.
    lane_annotations (torch.Tensor): Lane segmentation annotations.
    road_annotations (torch.Tensor): Road segmentation annotations.

    Returns:
    --------
    list of torch.Tensor
        If annotations are provided, returns a tuple of losses: (cls_loss, reg_loss, lane_loss, road_loss).
        If annotations are not provided, returns a tuple of detections (det_outputs, lane_output, road_output).
    """
    def __init__(self, config):
        super(HydraNet, self).__init__()
        self.regnetBackBone = RegNet(config["regnetConfig"])
        self.biFPN = BiFPN(config["biFPNConfig"])
        self.objectDetectionHead = ObjectDetectionHead(config["objectDetectionConfig"])
        self.segmentationHead = SegmentationHead(config["segmentationConfig"])
        self.detectionLoss = FocalLoss()
        self.laneSegLoss = FocalLossSeg(alpha=config["segmentationConfig"]["ALPHA_LANE"])
        self.roadSegLoss = FocalLossSeg(alpha=config["segmentationConfig"]["ALPHA_ROAD"])

    def forward(self, images, det_annotations=None, lane_annotations=None, road_annotations=None):
        stemFeatures, regnetFeatures = self.regnetBackBone(images)
        fusedFeatures = self.biFPN(regnetFeatures)
        classifications, regressions, anchors = self.objectDetectionHead(fusedFeatures, images)
        seg_outputs = self.segmentationHead(fusedFeatures, stemFeatures)
        lane_output = seg_outputs[:, 0, :, :]
        road_output = seg_outputs[:, 1, :, :]
        if det_annotations is not None:
            cls_loss, reg_loss = self.detectionLoss(classifications, regressions, anchors, det_annotations)
            lane_loss = self.laneSegLoss(lane_output.unsqueeze(1), lane_annotations)
            road_loss = self.roadSegLoss(road_output.unsqueeze(1), road_annotations)
            return (cls_loss, reg_loss, lane_loss, road_loss)
        else:
            det_outputs = self.objectDetectionHead.process_outputs(classifications, regressions, anchors, images)
            lane_output = torch.sigmoid(lane_output)
            road_output = torch.sigmoid(road_output)
            return (det_outputs, lane_output, road_output)
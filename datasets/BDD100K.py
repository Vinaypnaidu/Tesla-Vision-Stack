import os
import json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class BDD100KDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        """
        Initializes the BDD dataset.
        Parameters:
        - root_dir (str): Root directory of the dataset.
        - split (str): Type of dataset ('train' or 'val').
        """
        self.root_dir = root_dir
        self.split = split
        self.images_dir = os.path.join(root_dir, 'images', '100k', f'{split}')
        self.road_dir = os.path.join(root_dir, 'labels', 'drivable', 'colormaps', f'{split}')
        self.lane_dir = os.path.join(root_dir, 'labels', 'lane', 'colormaps', f'{split}')
        self.labels_path = os.path.join(root_dir, 'labels', 'det_20', f'det_{split}.json')

        with open(self.labels_path) as f:
            self.annotations = json.load(f)

        self.class_dict = {
            'pedestrian': 0,
            'rider': 1,
            'car': 2,
            'truck': 3,
            'bus': 4,
            'train': 5,
            'motorcycle': 6,
            'bicycle': 7,
            'traffic light': 8,
            'traffic sign': 9
        }

        self.valid_classes = {
            'pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
            'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
        }

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_info = self.annotations[idx]
        image_path = os.path.join(self.images_dir, image_info['name'])
        lane_path = os.path.join(self.lane_dir, image_info['name'].replace('.jpg', '.png'))
        road_path = os.path.join(self.road_dir, image_info['name'].replace('.jpg', '.png'))
        
        lane_label = cv2.imread(lane_path, 0)
        road_label = cv2.imread(road_path, 0)

        _, lane_label = cv2.threshold(lane_label, 0, 1, cv2.THRESH_BINARY)
        _, road_label = cv2.threshold(road_label, 0, 1, cv2.THRESH_BINARY)

        lane_label = np.expand_dims(lane_label, 2)
        road_label = np.expand_dims(road_label, 2)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        try:
            labels = image_info['labels']
        except KeyError:
            labels = []

        annotations = []
        for label in labels:
            category = label['category']
            if category in self.valid_classes:
                x1 = label['box2d']['x1']
                y1 = label['box2d']['y1']
                x2 = label['box2d']['x2']
                y2 = label['box2d']['y2']
                class_id = self.class_dict[category]
                annotations.append([x1, y1, x2, y2, class_id])
        
        annotations = np.array(annotations, dtype=np.float32) if annotations else np.empty((0, 5), dtype=np.float32)
        sample = {'images': image, 'det_annotations': annotations, 'lane_annotations': lane_label, 'road_annotations': road_label}
        return sample


def collater(data):
    imgs = [s['images'] for s in data]
    lane_labels = [s['lane_annotations'] for s in data]
    road_labels = [s['road_annotations'] for s in data]
    annots = [s['det_annotations'] for s in data]

    imgs = torch.stack([torch.from_numpy(img).permute(2, 0, 1) for img in imgs])
    lane_labels = torch.stack([torch.from_numpy(img).long().permute(2, 0, 1) for img in lane_labels])
    road_labels = torch.stack([torch.from_numpy(img).long().permute(2, 0, 1) for img in road_labels])

    max_num_annots = max(annot.shape[0] for annot in annots if annot.shape[0] > 0)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1  
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = torch.from_numpy(annot)
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1  
    
    return {'images': imgs, 'det_annotations': annot_padded, 'lane_annotations': lane_labels, 'road_annotations': road_labels}
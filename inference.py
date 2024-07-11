import os
import torch
import cv2
import numpy as np
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import yaml
from hydranet import HydraNet

def load_model(checkpoint_path, config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    model = HydraNet(config)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

def run_inference(model, image_folder, output_folder, detection_threshold=0.5, lane_threshold=0.5, road_threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # BGR
    class_dict = {
        0: ('pedestrian', (74, 181, 60)),
        1: ('rider', (211, 190, 250)),
        2: ('car', (75, 25, 229)),
        3: ('truck', (25, 225, 255)),
        4: ('bus', (230, 50, 240)),
        5: ('train', (254, 190, 221)),
        6: ('motorcycle', (49, 129, 245)),
        7: ('bicycle', (200, 130, 0)),
        8: ('traffic light', (240, 240, 70)),
        9: ('traffic sign', (180, 31, 144))
    }

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_path in tqdm(images, desc="Processing images"):
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = to_tensor(image_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            # Assume model returns scores, class IDs, and bounding boxes in this order
            det_outputs, lane_output, road_output = model(input_tensor)
            nms_scores, nms_classes, bboxes = det_outputs
            lane_output = lane_output.squeeze(0)
            road_output = road_output.squeeze(0)
        
        # Draw predictions on image
        for score, class_id, bbox in zip(nms_scores, nms_classes, bboxes):
            if score > detection_threshold:  # Apply threshold
                x1, y1, x2, y2 = bbox.int().cpu().numpy()
                class_name, color = class_dict.get(class_id.item(), ("Unknown", (255, 255, 255)))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f'{class_name}: {score:.2f}'
                
                # Improved text visibility
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, cv2.FILLED)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw road outputs
        road_output_np = road_output.detach().cpu().numpy()
        _, binary_mask = cv2.threshold(road_output_np, road_threshold, 1, cv2.THRESH_BINARY)
        binary_mask = np.uint8(binary_mask * 255)
        green_overlay = np.zeros_like(image)
        green_overlay[:, :] = (0, 255, 0)  # BGR format for green color
        green_masked_image = np.copy(image)
        green_masked_image[np.where(binary_mask == 255)] = green_overlay[np.where(binary_mask == 255)]
        image = cv2.addWeighted(image, 0.7, green_masked_image, 0.3, 0)

        # Draw lane outputs
        lane_output_np = lane_output.detach().cpu().numpy()
        _, binary_mask = cv2.threshold(lane_output_np, lane_threshold, 1, cv2.THRESH_BINARY)
        binary_mask = np.uint8(binary_mask * 255)
        red_overlay = np.zeros_like(image)
        red_overlay[:, :] = (0, 0, 255)  # BGR format for green color
        red_masked_image = np.copy(image)
        red_masked_image[np.where(binary_mask == 255)] = red_overlay[np.where(binary_mask == 255)]
        image = cv2.addWeighted(image, 0.7, red_masked_image, 0.3, 0)

        output_path = os.path.join(output_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, image)

    print(f"Processed {len(images)} images and saved to {output_folder}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a folder of images")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the corresponding model config")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing images for inference")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save output images")
    parser.add_argument("--det_threshold", type=float, default=0.5, help="Detection threshold for bounding boxes")
    parser.add_argument("--lane_threshold", type=float, default=0.5, help="Detection threshold for lane segmentation")
    parser.add_argument("--road_threshold", type=float, default=0.5, help="Detection threshold for road segmentation")
    args = parser.parse_args()

    model = load_model(args.checkpoint_path, args.config_path)
    run_inference(model, args.image_folder, args.output_folder, args.det_threshold, args.lane_threshold, args.road_threshold)
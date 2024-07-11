import os
import argparse
import shutil
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm
from hydranet import HydraNet
from datasets.BDD100K import BDD100KDataset, collater

def get_args():
    parser = argparse.ArgumentParser("HydraNet Implementation by Vinay Purushotham")
    parser.add_argument("--batch_size", type=int, default=8, help="The number of images per batch")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--test_interval", type=int, default=1)
    parser.add_argument("--config_path", type=str, default="config.yaml", help="path to HydraNet config.yaml")
    parser.add_argument("--data_path", type=str, default="data/BDD100K", help="the root folder of dataset")
    parser.add_argument("--log_path", type=str, default="runs/run1/logs")
    parser.add_argument("--save_path", type=str, default="runs/run1/trained_models")
    parser.add_argument("--resume_path", type=str, default=None, help="path to pretrained model")
    parser.add_argument("--freeze_backbone", type=bool, default=False, help="freeze backbone")
    parser.add_argument("--freeze_det", type=bool, default=False, help="freeze detection head")
    parser.add_argument("--freeze_seg", type=bool, default=False, help="freeze segmentation head")
    parser.add_argument("--cls_weight", type=float, default=1.0)
    parser.add_argument("--reg_weight", type=float, default=1.0)
    parser.add_argument("--lane_weight", type=float, default=25.0)
    parser.add_argument("--road_weight", type=float, default=10.0)
    args = parser.parse_args()
    return args

def train(opt):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(2109)
    else:
        device = torch.device("cpu")
        torch.manual_seed(2109)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)

    train_params = {"batch_size": opt.batch_size,
                    "shuffle": True,
                    "drop_last": True,
                    "num_workers": opt.num_workers,
                    "collate_fn": collater}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False,
                    "num_workers": opt.num_workers,
                    "collate_fn": collater}

    train_set = BDD100KDataset(opt.data_path, split='train')
    train_loader = DataLoader(train_set, **train_params)
    val_set = BDD100KDataset(root_dir=opt.data_path, split='val')
    val_loader = DataLoader(val_set, **test_params)

    with open(opt.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    print(f"Hydranet config path: {opt.config_path}")
    model = HydraNet(config)

    if opt.resume_path != None:
        print(f"Loading pretrained weights from: {opt.resume_path}")
        model.load_state_dict(torch.load(opt.resume_path))

    if opt.freeze_backbone:
        print('Freezing backbone...')
        model.regnetBackBone.requires_grad_(False)
        model.biFPN.requires_grad_(False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    
    print(f"Learning rate: {opt.lr}")
    print(f"Training started on {device}!")
    step = 0
    for epoch in range(1, opt.num_epochs + 1):
        model.train()
        epoch_loss = []
        epoch_cls_loss = []
        epoch_reg_loss = []
        epoch_lane_loss = []
        epoch_road_loss = []

        progress_bar = tqdm(train_loader)
        for iter, data in enumerate(progress_bar):
            optimizer.zero_grad()
            images = data['images'].float()
            det_annotations = data['det_annotations']
            lane_annotations = data['lane_annotations']
            road_annotations = data['road_annotations']
            images = images.to(device)
            det_annotations = det_annotations.to(device)
            lane_annotations = lane_annotations.to(device)
            road_annotations = road_annotations.to(device)

            cls_loss, reg_loss, lane_loss, road_loss = model(images, det_annotations, lane_annotations, road_annotations)
            cls_loss = cls_loss.mean() if not opt.freeze_det else torch.tensor(0, device=device) * opt.cls_weight
            reg_loss = reg_loss.mean() if not opt.freeze_det else torch.tensor(0, device=device) * opt.reg_weight
            lane_loss = (lane_loss.mean() if not opt.freeze_seg else torch.tensor(0, device=device)) * opt.lane_weight
            road_loss = (road_loss.mean() if not opt.freeze_seg else torch.tensor(0, device=device)) * opt.road_weight
            loss = cls_loss + reg_loss + lane_loss + road_loss

            if loss == 0:
                continue

            loss.backward()
            optimizer.step()

            epoch_loss.append(float(loss.item()))
            epoch_cls_loss.append(float(cls_loss.item()))
            epoch_reg_loss.append(float(reg_loss.item()))
            epoch_lane_loss.append(float(lane_loss.item()))
            epoch_road_loss.append(float(road_loss.item()))

            total_loss = np.mean(epoch_loss)
            total_cls_loss = np.mean(epoch_cls_loss)
            total_reg_loss = np.mean(epoch_reg_loss)
            total_lane_loss = np.mean(epoch_lane_loss)
            total_road_loss = np.mean(epoch_road_loss)

            progress_bar.set_description(
                'Epoch: {}/{}. Total loss: {:.5f} Cls: {:.5f} Reg: {:.5f} Lane: {:.5f} Road: {:.5f}'.format(
                 epoch, opt.num_epochs, total_loss, total_cls_loss, total_reg_loss, total_lane_loss, total_road_loss))

            writer.add_scalar('Train/Total_loss', total_loss, step)
            writer.add_scalar('Train/Cls_loss', total_cls_loss, step)
            writer.add_scalar('Train/Reg_loss', total_reg_loss, step)
            writer.add_scalar('Train/Lane_loss', total_lane_loss, step)
            writer.add_scalar('Train/Road_loss', total_road_loss, step)
            step += 1

        if epoch % opt.test_interval == 0:
            model.eval()
            losses = []
            cls_losses = []
            reg_losses = []
            lane_losses = []
            road_losses = []

            for iter, data in enumerate(val_loader):
                with torch.no_grad():
                    images = data['images'].float()
                    det_annotations = data['det_annotations']
                    lane_annotations = data['lane_annotations']
                    road_annotations = data['road_annotations']
                    images = images.to(device)
                    det_annotations = det_annotations.to(device)
                    lane_annotations = lane_annotations.to(device)
                    road_annotations = road_annotations.to(device)

                    cls_loss, reg_loss, lane_loss, road_loss = model(images, det_annotations, lane_annotations, road_annotations)
                    cls_loss = cls_loss.mean() * opt.cls_weight
                    reg_loss = reg_loss.mean() * opt.reg_weight
                    lane_loss = lane_loss.mean() * opt.lane_weight
                    road_loss = road_loss.mean() * opt.road_weight
                    loss = cls_loss + reg_loss + lane_loss + road_loss

                    losses.append(float(loss.item()))
                    cls_losses.append(float(cls_loss.item()))
                    reg_losses.append(float(reg_loss.item()))
                    lane_losses.append(float(lane_loss.item()))
                    road_losses.append(float(road_loss.item()))

            loss = np.mean(losses)
            cls_loss = np.mean(cls_losses)
            reg_loss = np.mean(reg_losses)
            lane_loss = np.mean(lane_losses)
            road_loss = np.mean(road_losses)

            print('Epoch: {}/{}. Total loss: {:1.5f} Cls: {:.5f} Reg: {:.5f} Lane: {:.5f} Road: {:.5f}'.format(
                  epoch, opt.num_epochs, loss, cls_loss, reg_loss, lane_loss, road_loss))
            
            writer.add_scalar('Val/Total_loss', loss, epoch)
            writer.add_scalar('Val/Cls_loss', cls_loss, epoch)
            writer.add_scalar('Val/Reg_loss', reg_loss, epoch)
            writer.add_scalar('Val/Lane_loss', lane_loss, epoch)
            writer.add_scalar('Val/Road_loss', road_loss, epoch)

        save_path = f"{opt.save_path}/weights_epoch_{epoch}.pth"
        print(f"Model saved to {save_path}")
        torch.save(model.state_dict(), save_path)
        
    print("Training completed.")
    writer.close()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
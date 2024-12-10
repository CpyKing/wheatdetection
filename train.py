import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from matplotlib import pyplot as plt

from dataloader import WheatDataset
from tools import *
from models import *


if __name__ == '__main__':

    DIR_INPUT = '/workspace/my_wheat'
    DIR_TRAIN = f'{DIR_INPUT}/dataset/train'
    DIR_TEST = f'{DIR_INPUT}/dataset/test'

    train_df = pd.read_csv(f'{DIR_INPUT}/dataset/train.csv')
    print(train_df.shape)

    train_df['x'] = -1
    train_df['y'] = -1
    train_df['w'] = -1
    train_df['h'] = -1

    train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
    train_df.drop(columns=['bbox'], inplace=True)
    train_df['x'] = train_df['x'].astype(float)
    train_df['y'] = train_df['y'].astype(float)
    train_df['w'] = train_df['w'].astype(float)
    train_df['h'] = train_df['h'].astype(float)

    image_ids = train_df['image_id'].unique()
    valid_ids = image_ids[-665:]
    train_ids = image_ids[:-665]

    valid_df = train_df[train_df['image_id'].isin(valid_ids)]
    train_df = train_df[train_df['image_id'].isin(train_ids)]

    print(valid_df.shape, train_df.shape)


    model = get_model()

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_dataset = WheatDataset(train_df, DIR_TRAIN, get_train_transform(), 'train')
    valid_dataset = WheatDataset(valid_df, DIR_TRAIN, get_valid_transform(), 'train')

    # split the dataset in train and test set
    indices = torch.randperm(len(train_dataset)).tolist()

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    images, targets, image_ids = next(iter(train_data_loader))
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    # lr_scheduler = None

    num_epochs = 2

    loss_hist = Averager()
    itr = 1

    for epoch in range(num_epochs):
        loss_hist.reset()
        
        for images, targets, image_ids in train_data_loader:
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 2 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1
        
        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        print(f"Epoch #{epoch} loss: {loss_hist.value}")


    torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')

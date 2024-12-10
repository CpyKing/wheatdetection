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
from dataloader import *
from tools import get_test_transform
from models import get_model

DIR_INPUT = '/workspace/my_wheat'
DIR_TRAIN = f'{DIR_INPUT}/dataset/train'
DIR_TEST = f'{DIR_INPUT}/dataset/test'

DIR_WEIGHTS = '/workspace/my_wheat'

WEIGHTS_FILE = f'{DIR_WEIGHTS}/fasterrcnn_resnet50_fpn_best.pth'

test_df = pd.read_csv(f'{DIR_INPUT}/dataset/submission.csv')
print(test_df.shape)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model()
# Load the trained weights
model.load_state_dict(torch.load(WEIGHTS_FILE))
model.eval()

x = model.to(device)

def collate_fn(batch):
    return tuple(zip(*batch))

test_dataset = WheatDataset(test_df, DIR_TEST, get_test_transform(), 'test')

test_data_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    drop_last=False,
    collate_fn=collate_fn
)

detection_threshold = 0.5
results = []

for images, image_ids in test_data_loader:

    images = list(image.to(device) for image in images)
    outputs = model(images)

    for i, image in enumerate(images):

        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]
        
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        

        for j in zip(scores, boxes):
            results.append({
                'image_id': image_id,
                'PredictionString': "{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3])
            })

print(results[0:2])

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
print(test_df.head())

sample = images[1].permute(1,2,0).cpu().numpy()
boxes = outputs[1]['boxes'].data.cpu().numpy()
scores = outputs[1]['scores'].data.cpu().numpy()

boxes = boxes[scores >= detection_threshold].astype(np.int32)

fig, ax = plt.subplots(1, 1, figsize=(16, 8))

for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (220, 0, 0), 2)
    
ax.set_axis_off()
ax.imshow(sample)
plt.savefig('inf.jpg')



test_df.to_csv('submission.csv', index=False)
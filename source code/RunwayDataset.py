import os, json, glob, cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import timm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, gc, random, torchvision
from torchvision.ops import nms, batched_nms
import pandas as pd
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.empty_cache()

# =========================================================
# 1.  DATASET  (YOLO TXT layout)
# =========================================================
class RunwayDataset(Dataset):
    def __init__(self, root, split='train', img_size=512):
        self.root = root
        self.split = split
        self.img_size = (img_size, img_size)
        img_dir = os.path.join(root, 'images', split)
        lbl_dir = os.path.join(root, 'labels', split)
        assert os.path.exists(img_dir), f"Image directory not found: {img_dir}"
        assert os.path.exists(lbl_dir), f"Label directory not found: {lbl_dir}"

        self.samples = []
        for img_path in glob.glob(os.path.join(img_dir, '*')):
            name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, name + '.txt')
            if os.path.exists(lbl_path):
                self.samples.append((img_path, lbl_path))
        print(f'{split}: {len(self.samples)} samples')

        if split == 'train':
            self.tf = A.Compose([
                A.Resize(*self.img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['cls']))
        else:
            self.tf = A.Compose([
                A.Resize(*self.img_size),
                A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['cls']))

    def parse_yolo(self, path):
        boxes, labels = [], []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = map(float, parts)
                boxes.append([cx, cy, w, h])
                labels.append(int(cls))
        return boxes, labels

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        boxes, labels = self.parse_yolo(lbl_path)
        
        if not boxes:
            boxes = []
            labels = []
        
        tf = self.tf(image=img, bboxes=boxes, cls=labels)
        img, boxes, labels = tf['image'], tf['bboxes'], tf['cls']
        
        h, w = self.img_size
        targets = []
        for (cx, cy, wb, hb), l in zip(boxes, labels):
            x1 = max(0, (cx - wb/2) * w)
            y1 = max(0, (cy - hb/2) * h)
            x2 = min(w, (cx + wb/2) * w)
            y2 = min(h, (cy + hb/2) * h)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            targets.append([x1, y1, x2, y2, l])
        
        targets = torch.as_tensor(targets, dtype=torch.float32) if targets else torch.empty((0, 5))
        return img, targets

def collate(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), targets

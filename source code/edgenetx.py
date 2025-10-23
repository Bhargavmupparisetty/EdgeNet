
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
import RunwayDataset, Dynamic_NMS, Dynamic_head, custom_BiFPN



class EdgeAirportNet(nn.Module):
    def __init__(self, num_classes=2, img_size=512):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model('ghostnetv2_100', pretrained=True, features_only=True)
        ch = self.backbone.feature_info.channels()[1:]  # [40, 112, 160, 960]
        self.bifpn = BiFPN(ch, 128)
        
        # Dynamic head instead of regular head
        self.head = DynamicHead(128, num_classes, num_levels=4)
        
        # Dynamic NMS
        self.nms = DynamicNMS(num_classes, score_threshold=0.3, max_detections=300)
        
        # Omega gates for each level
        self.omega_g = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 128),
                nn.Sigmoid()
            ) for _ in range(4)
        ])
        
    def forward(self, x, apply_nms=False):
        feats = self.backbone(x)[1:]  
        fpn_feats = self.bifpn(feats)
        
        # Apply omega gates
        gated_feats = []
        for i, f in enumerate(fpn_feats):
            g = self.omega_g[i](F.adaptive_avg_pool2d(f, 1)).view(-1, 128, 1, 1)
            gated_feats.append(f * g)
            
        cls_logits, bbox_preds = self.head(gated_feats)
        
        if apply_nms:
            return self.post_process(cls_logits, bbox_preds, x.size(-1))
        
        return cls_logits, bbox_preds
    
    def post_process(self, cls_logits, bbox_preds, img_size):
        """Post-process predictions with dynamic NMS"""
        B = cls_logits[0].size(0)
        strides = [4, 8, 16, 32]
        
        all_boxes, all_scores, all_labels = [], [], []
        
        for cls_pred, reg_pred, stride in zip(cls_logits, bbox_preds, strides):
            _, C, H, W = cls_pred.shape
            
            # Reshape predictions
            cls_pred = torch.sigmoid(cls_pred.permute(0, 2, 3, 1).reshape(B, -1, C))
            reg_pred = reg_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
            
            # Generate anchor points
            shift_x = torch.arange(0, W, dtype=torch.float32, device=cls_pred.device) * stride
            shift_y = torch.arange(0, H, dtype=torch.float32, device=cls_pred.device) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            
            # Convert to boxes
            x1 = shift_x - reg_pred[..., 0]
            y1 = shift_y - reg_pred[..., 1]
            x2 = shift_x + reg_pred[..., 2]
            y2 = shift_y + reg_pred[..., 3]
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            
            scores, labels = cls_pred.max(dim=-1)
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
        
        # Concatenate all levels
        all_boxes = torch.cat(all_boxes, dim=1)
        all_scores = torch.cat(all_scores, dim=1)
        all_labels = torch.cat(all_labels, dim=1)
        
        # Apply dynamic NMS
        return self.nms(all_boxes, all_scores, all_labels, img_size)

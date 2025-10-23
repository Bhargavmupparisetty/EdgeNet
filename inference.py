# =========================================================
#   ENV & IMPORTS
# =========================================================

import sys
import subprocess
import importlib

packages = {
    "timm": "timm",
    "albumentations": "albumentations",
    "torchmetrics": "torchmetrics"
}

for pkg_name, import_name in packages.items():
    try:
        importlib.import_module(import_name)
        print(f"{pkg_name} is already installed.")
    except ImportError:
        print(f"Installing {pkg_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        print(f"{pkg_name} installed successfully.")

print("\nAll required packages are now available.")


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
#   DATASET  (YOLO TXT layout)
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

# =========================================================
#   DYNAMIC NMS MODULE
# =========================================================
class DynamicNMS(nn.Module):
    def __init__(self, num_classes=2, score_threshold=0.3, max_detections=300):
        super().__init__()
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        
        # Learnable NMS parameters
        self.nms_thresh = nn.Parameter(torch.tensor(-0.6))
        self.class_weights = nn.Parameter(torch.ones(num_classes))
        
    def forward(self, boxes, scores, labels, img_size=512):
        batch_size = boxes.size(0)
        results = []
        
        for b in range(batch_size):
            # Get valid detections above threshold
            valid_mask = scores[b] > self.score_threshold
            
            if not valid_mask.any():
                results.append({
                    'boxes': torch.empty((0, 4), device=boxes.device),
                    'scores': torch.empty(0, device=boxes.device),
                    'labels': torch.empty(0, dtype=torch.long, device=boxes.device)
                })
                continue
            
            valid_boxes = boxes[b][valid_mask]
            valid_scores = scores[b][valid_mask]
            valid_labels = labels[b][valid_mask]
            
            # Apply class-specific score weighting
            weighted_scores = valid_scores * self.class_weights[valid_labels]
            
            # Clip boxes to image bounds
            valid_boxes[:, 0] = torch.clamp(valid_boxes[:, 0], 0, img_size)
            valid_boxes[:, 1] = torch.clamp(valid_boxes[:, 1], 0, img_size)
            valid_boxes[:, 2] = torch.clamp(valid_boxes[:, 2], 0, img_size)
            valid_boxes[:, 3] = torch.clamp(valid_boxes[:, 3], 0, img_size)
            
            # Apply batched NMS with learnable threshold
            nms_threshold = torch.sigmoid(self.nms_thresh)  # Keep in [0,1]
            keep_indices = batched_nms(valid_boxes, weighted_scores, valid_labels, nms_threshold)
            
            # Limit to max detections
            if len(keep_indices) > self.max_detections:
                keep_indices = keep_indices[:self.max_detections]
            
            results.append({
                'boxes': valid_boxes[keep_indices],
                'scores': valid_scores[keep_indices],  # Use original scores
                'labels': valid_labels[keep_indices]
            })
        
        return results

# =========================================================
#   DYNAMIC HEAD MODULE
# =========================================================
class DynamicHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_levels=4):
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = num_levels
        
        # Shared convolutions
        self.shared_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        # Dynamic weight generation networks
        self.cls_weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels * num_classes, 1)
        )
        
        self.reg_weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels * 4, 1)
        )
        
        # Level-specific attention
        self.level_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, num_levels, 1),
            nn.Softmax(dim=1)
        )
        
        # Final prediction layers
        self.cls_final = nn.Conv2d(in_channels, num_classes, 1)
        self.reg_final = nn.Conv2d(in_channels, 4, 1)
        
    def forward(self, features):
        cls_logits = []
        bbox_preds = []
        
        for i, feat in enumerate(features):
            B, C, H, W = feat.shape
            
            # Shared feature processing
            shared_feat = F.relu(self.shared_conv(feat))
            
            # Generate dynamic weights
            cls_weights = self.cls_weight_net(shared_feat)  # (B, C*num_classes, 1, 1)
            reg_weights = self.reg_weight_net(shared_feat)   # (B, C*4, 1, 1)
            
            cls_weights = cls_weights.view(B, C, self.num_classes, 1, 1)
            reg_weights = reg_weights.view(B, C, 4, 1, 1)
            
            # Apply dynamic convolution
            shared_feat_expanded = shared_feat.unsqueeze(2)  # (B, C, 1, H, W)
            
            # Classification branch
            cls_feat = (shared_feat_expanded * cls_weights).sum(dim=1)  # (B, num_classes, H, W)
            cls_out = self.cls_final(shared_feat) + cls_feat
            
            # Regression branch
            reg_feat = (shared_feat_expanded * reg_weights).sum(dim=1)  # (B, 4, H, W)
            reg_out = self.reg_final(shared_feat) + reg_feat
            
            # Apply level attention
            level_att = self.level_attention(shared_feat)[:, i:i+1, :, :]  # (B, 1, 1, 1)
            
            cls_logits.append(cls_out * level_att)
            bbox_preds.append(reg_out * level_att)
        
        return cls_logits, bbox_preds

# =========================================================
#   ENHANCED MODEL WITH DYNAMIC COMPONENTS
# =========================================================
class BiFPN(nn.Module):
    def __init__(self, ch_ins, out_ch):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in ch_ins])
        self.smooth = nn.ModuleList([nn.Conv2d(out_ch, out_ch, 3, padding=1) for _ in range(len(ch_ins))])
        
    def forward(self, feats):
        laterals = [lat(f) for lat, f in zip(self.lateral, feats)]
        p2, p3, p4, p5 = laterals
        
        # Top-down pathway
        p4 = p4 + F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p3 = p3 + F.interpolate(p4, size=p3.shape[-2:], mode='nearest')
        p2 = p2 + F.interpolate(p3, size=p2.shape[-2:], mode='nearest')
        
        # Smooth
        p2 = self.smooth[0](p2)
        p3 = self.smooth[1](p3)
        p4 = self.smooth[2](p4)
        p5 = self.smooth[3](p5)
        
        return [p2, p3, p4, p5]

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
        feats = self.backbone(x)[1:]  # Drop stem, keep 4 levels
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

# =========================================================
#   ENHANCED INFERENCE WITH DETAILED VISUALIZATIONS
# =========================================================
def inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EdgeAirportNet().to(device)

    # Load model weights
    model_path = 'models/airport_detection.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded model weights from", model_path)
    else:
        print("Model weights not found. Using random initialization.")
    
    model.eval()

    # Load dataset - corrected to use 'val' split properly
    val_dataset = RunwayDataset('sample_dataset', split='val', img_size=512)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=2, collate_fn=collate)

    metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75])
    all_confidences, all_box_areas, all_predictions = [], [], []

    # Setup plots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    sample_count = 0

    with torch.no_grad():
        for imgs, targets in val_loader:
            if sample_count >= 12:
                break

            imgs = imgs.to(device)
            predictions = model.post_process(*model(imgs), img_size=512)

            # Format targets for metric - corrected to handle the dataset format
            formatted_targets = []
            for target in targets:
                if target.numel() == 0:
                    formatted_targets.append({
                        'boxes': torch.empty((0, 4), device=device),
                        'labels': torch.empty(0, dtype=torch.long, device=device)
                    })
                else:
                    # Target format is [x1, y1, x2, y2, label]
                    formatted_targets.append({
                        'boxes': target[:, :4].to(device),  # First 4 columns are box coordinates
                        'labels': target[:, 4].long().to(device)  # Last column is label
                    })

            metric.update(predictions, formatted_targets)

            # Visualize
            for b in range(min(imgs.size(0), 12 - sample_count)):
                img = imgs[b].cpu().permute(1, 2, 0).numpy()
                # Corrected denormalization - std first, then mean
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)

                axes[sample_count].imshow(img)
                axes[sample_count].axis('off')

                # Predicted boxes
                pred = predictions[b]
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    if score < 0.5:
                        continue
                    x1, y1, x2, y2 = box
                    color = 'red' if score > 0.7 else 'orange'
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=2, edgecolor=color, facecolor='none')
                    axes[sample_count].add_patch(rect)
                    axes[sample_count].text(x1, y1 - 5, f'{score:.2f}',
                                            color=color, fontsize=8, weight='bold')

                # Ground truth boxes - corrected to handle dataset format
                if targets[b].numel() > 0:
                    # targets[b] format is [x1, y1, x2, y2, label]
                    gt_boxes = targets[b][:, :4].cpu().numpy()  # Extract first 4 columns
                    for box in gt_boxes:
                        x1, y1, x2, y2 = box
                        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                             linewidth=2, edgecolor='green', facecolor='none')
                        axes[sample_count].add_patch(rect)

                # Count ground truth boxes properly
                gt_count = len(targets[b]) if targets[b].numel() > 0 else 0
                axes[sample_count].set_title(f"Sample {sample_count+1}\nPred: {len(boxes)} | GT: {gt_count}")

                all_confidences.extend(scores)
                all_box_areas.extend([(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes])
                all_predictions.append(len(boxes))
                sample_count += 1

    plt.tight_layout()
    plt.show()

    # Final metrics
    m = metric.compute()

    # Stats plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Handle case where no predictions were made
    if not all_confidences:
        print("Warning: No predictions made with confidence > 0.5")
        all_confidences = [0]
        all_box_areas = [0]

    # Histogram: Confidence
    axes[0, 0].hist(all_confidences, bins=50, color='skyblue')
    axes[0, 0].set_title("Confidence Distribution")
    axes[0, 0].set_xlabel("Confidence")
    axes[0, 0].grid(True)

    # Histogram: Box Area
    axes[0, 1].hist(all_box_areas, bins=50, color='lightgreen')
    axes[0, 1].set_title("Box Area Distribution")
    axes[0, 1].set_xlabel("Area (px²)")
    axes[0, 1].grid(True)

    # Histogram: Predictions per image
    axes[0, 2].hist(all_predictions, bins=20, color='salmon')
    axes[0, 2].set_title("Predictions per Image")
    axes[0, 2].set_xlabel("Count")
    axes[0, 2].grid(True)

    # Metrics Bar Plot
    metric_keys = ['map_50', 'map_75', 'map', 'mar_100']
    metric_vals = [m[k].item() for k in metric_keys]
    bars = axes[1, 0].bar(metric_keys, metric_vals, color='mediumseagreen')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Validation Metrics")
    for bar, val in zip(bars, metric_vals):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}",
                        ha='center', va='bottom')

    # Scatter: Confidence vs Area
    if len(all_confidences) > 1 and len(all_box_areas) > 1:
        axes[1, 1].scatter(all_box_areas, all_confidences, alpha=0.6, c='purple')
    axes[1, 1].set_title("Confidence vs Box Area")
    axes[1, 1].set_xlabel("Area")
    axes[1, 1].set_ylabel("Confidence")
    axes[1, 1].grid(True)

    # Summary Text
    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.8, f"Total Predictions: {len(all_confidences)}", fontsize=12)
    if all_confidences and all_confidences != [0]:
        axes[1, 2].text(0.1, 0.7, f"Avg Confidence: {np.mean(all_confidences):.3f}", fontsize=12)
        axes[1, 2].text(0.1, 0.6, f"Avg Box Area: {np.mean(all_box_areas):.1f}", fontsize=12)
    else:
        axes[1, 2].text(0.1, 0.7, "Avg Confidence: N/A", fontsize=12)
        axes[1, 2].text(0.1, 0.6, "Avg Box Area: N/A", fontsize=12)
    axes[1, 2].text(0.1, 0.5, f"Avg Preds/Image: {np.mean(all_predictions):.1f}", fontsize=12)
    axes[1, 2].set_title("Performance Summary")

    plt.tight_layout()
    plt.show()

    return m

# Run inference directly
if __name__ == '__main__':
    print("Starting inference...")
    final_metrics = inference()

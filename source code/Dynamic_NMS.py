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
        """
        boxes: (B, N, 4) - predicted boxes
        scores: (B, N) - confidence scores
        labels: (B, N) - predicted labels
        """
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
                'scores': valid_scores[keep_indices],  
                'labels': valid_labels[keep_indices]
            })
        
        return results

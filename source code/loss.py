# =========================================================
#   ENHANCED LOSS FUNCTION
# =========================================================
def compute_loss(cls_logits, bbox_preds, targets, img_size=512):
    device = cls_logits[0].device
    strides = [4, 8, 16, 32]
    
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    total_pos_samples = 0
    
    for level, (cls_pred, reg_pred, stride) in enumerate(zip(cls_logits, bbox_preds, strides)):
        B, C, H, W = cls_pred.shape
        
        cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(B, -1, C)
        reg_pred = reg_pred.permute(0, 2, 3, 1).reshape(B, -1, 4)
        
        shift_x = torch.arange(0, W, dtype=torch.float32, device=device) * stride
        shift_y = torch.arange(0, H, dtype=torch.float32, device=device) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack([shift_x, shift_y], dim=1)
        
        for b in range(B):
            gt_boxes = targets[b]
            
            if gt_boxes.numel() == 0:
                cls_target = torch.zeros((H * W, C), device=device, dtype=torch.float32)
                neg_loss = F.binary_cross_entropy_with_logits(
                    cls_pred[b], cls_target, reduction='sum'
                )
                total_cls_loss += neg_loss / (H * W)
                continue
            
            gt_boxes = gt_boxes.to(device)
            gt_classes = gt_boxes[:, 4].long()
            gt_boxes_xyxy = gt_boxes[:, :4]
            
            gt_cx = (gt_boxes_xyxy[:, 0] + gt_boxes_xyxy[:, 2]) / 2
            gt_cy = (gt_boxes_xyxy[:, 1] + gt_boxes_xyxy[:, 3]) / 2
            gt_centers = torch.stack([gt_cx, gt_cy], dim=1)
            
            distances = torch.cdist(locations, gt_centers)
            min_distances, closest_gt_idx = distances.min(dim=1)
            
            distance_threshold = stride * 1.5
            pos_mask = min_distances < distance_threshold
            
            cls_target = torch.zeros((H * W, C), device=device, dtype=torch.float32)
            if pos_mask.any():
                pos_locations = torch.where(pos_mask)[0]
                pos_gt_classes = gt_classes[closest_gt_idx[pos_mask]]
                cls_target[pos_locations, pos_gt_classes] = 1.0
            
            cls_loss = torchvision.ops.sigmoid_focal_loss(
                cls_pred[b], cls_target, alpha=0.25, gamma=2.0, reduction='sum'
            )
            total_cls_loss += cls_loss / max(pos_mask.sum().item(), 1)
            
            if pos_mask.any():
                pos_reg_pred = reg_pred[b][pos_mask]
                pos_locations_xy = locations[pos_mask]
                pos_gt_boxes = gt_boxes_xyxy[closest_gt_idx[pos_mask]]
                
                pred_x1 = pos_locations_xy[:, 0] - pos_reg_pred[:, 0]
                pred_y1 = pos_locations_xy[:, 1] - pos_reg_pred[:, 1]
                pred_x2 = pos_locations_xy[:, 0] + pos_reg_pred[:, 2]
                pred_y2 = pos_locations_xy[:, 1] + pos_reg_pred[:, 3]
                pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
                
                iou_loss = torchvision.ops.complete_box_iou_loss(
                    pred_boxes, pos_gt_boxes, reduction='mean'
                )
                total_reg_loss += iou_loss
                total_pos_samples += pos_mask.sum().item()
    
    if total_pos_samples > 0:
        total_reg_loss = total_reg_loss / len(strides)
    else:
        total_reg_loss = torch.tensor(0.0, device=device)
    
    total_cls_loss = total_cls_loss / (len(strides) * B)
    
    return total_cls_loss + total_reg_loss

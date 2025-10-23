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

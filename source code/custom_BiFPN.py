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

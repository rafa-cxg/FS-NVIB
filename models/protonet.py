import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # bias & scale of cosine classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

        # backbone
        self.backbone = backbone

    def cos_classifier(self, w, f):
        """
        w.shape = B, nC, d
        f.shape = B, M, d
        """
        f = F.normalize(f, p=2, dim=f.dim()-1, eps=1e-12)
        w = F.normalize(w, p=2, dim=w.dim()-1, eps=1e-12)

        cls_scores = f @ w.transpose(1, 2) # B, M, nC
        cls_scores = self.scale_cls * (cls_scores + self.bias)
        return cls_scores

    def forward(self, supp_x, supp_y, x, set_mata_training_mode=False):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        num_classes = supp_y.max() + 1 # NOTE: assume B==1

        B, nSupp, C, H, W = supp_x.shape
        # check if NVIB model
        if  self.backbone.__class__.__name__ == "NvibVisionTransformer":

            supp_f, (klg_supp, kld_supp) = self.backbone.forward(supp_x.view(-1, C, H, W),set_mata_training_mode=set_mata_training_mode)
        else: 
            supp_f = self.backbone.forward(supp_x.view(-1, C, H, W))
        supp_f = supp_f.view(B, nSupp, -1)

        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2) # B, nC, nSupp

        # B, nC, nSupp x B, nSupp, d = B, nC, d
        prototypes = torch.bmm(supp_y_1hot.float(), supp_f)
        prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0 if some classes got 0 images

        # Get KL diveregence from the feat
        if  self.backbone.__class__.__name__ == "NvibVisionTransformer":
            feat, (klg, kld) = self.backbone.forward(x.view(-1, C, H, W),set_mata_training_mode=set_mata_training_mode)
        else: 
            feat = self.backbone.forward(x.view(-1, C, H, W))

        feat = feat.view(B, x.shape[1], -1) # B, nQry, d

        logits = self.cos_classifier(prototypes, feat) # B, nQry, nC
        if  self.backbone.__class__.__name__ == "NvibVisionTransformer":
            return logits, (klg+klg_supp, kld+kld_supp)
        else:
            return logits

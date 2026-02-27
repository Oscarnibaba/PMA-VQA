from collections import OrderedDict
import sys
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel


class _LAVTVQA(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask):
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        # Check if classifier needs text features (for TextGuidedVQAHead)
        if hasattr(self.classifier, '_needs_text') and self.classifier._needs_text:
            return self.classifier(x_c4, l_feats, l_mask)
        # Check if classifier supports multi-scale features (for PFFCM)
        elif hasattr(self.classifier, 'fusion_conv'):
            return self.classifier((x_c1, x_c2, x_c3, x_c4))
        # Default: single-scale C4 only (for SimpleVQAHead)
        else:
            return self.classifier(x_c4)


class LAVTVQA(_LAVTVQA):
    pass

###############################################
# LAVT One VQA: put BERT inside the overall model #
###############################################
class _LAVTVQAOne(nn.Module):
    def __init__(self, backbone, classifier, args):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

    def forward(self, x, text, l_mask):
        ### language inference ###
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # (B, N_l, 768)
        l_feats = l_feats.permute(0, 2, 1)  # (B, 768, N_l) to make Conv1d happy
        l_mask = l_mask.unsqueeze(dim=-1)  # (batch, N_l, 1)
        ##########################
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features
        # 如果分类头是多尺度融合头（不需要文本输入），直接传递多尺度特征
        if hasattr(self.classifier, 'fusion_conv'):
            return self.classifier((x_c1, x_c2, x_c3, x_c4))
        # 否则按原先接口传入 c4 与文本特征
        return self.classifier(x_c4, l_feats, l_mask)


class LAVTVQAOne(_LAVTVQAOne):
    pass


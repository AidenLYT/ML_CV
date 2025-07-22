import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict


class R34_ver1(nn.Module):
    def __init__(
            self,
            num_classes: int = 50,
            dropout_p: float = 0.3,
            freeze_backbone: bool = False,
    ):
        super(R34_ver1, self).__init__()

        resnet = models.resnet34(pretrained=True)
        backbone = nn.Sequential(OrderedDict([*(list(resnet.named_children())[:-2])]))

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes),
        )

    def forward(self, img):
        feat_map = self.backbone(img)
        feat_1d = self.avgpool(feat_map).flatten(1)
        logit = self.classifier(feat_1d)

        return logit

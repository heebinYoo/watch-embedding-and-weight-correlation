import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50


class ConfidenceControl(nn.Module):
    def __init__(self, feature_dim, number_of_class, model_type, set_as_normal_classifier=False):
        super(ConfidenceControl, self).__init__()
        if model_type=="resnet":
            self.feature_extractor = ResnetFeatureExtractor(feature_dim)
        elif model_type =="simple":
            self.feature_extractor = SimpleFeatureExtractor(feature_dim)


        if set_as_normal_classifier:
            self.classifier = nn.Linear(feature_dim, number_of_class)
        else:
            self.classifier = nn.Linear(feature_dim, number_of_class * 2)


    def forward(self, x):
        t = self.feature_extractor(x)
        return self.classifier(t)

    def forward_feature(self, x):
        return self.feature_extractor(x)


class ResnetFeatureExtractor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.feature = []
        for name, module in resnet50(pretrained=True).named_children():
            if isinstance(module, nn.Linear):
                continue
            self.feature.append(module)
        self.feature = nn.Sequential(*self.feature)

        # Refactor Layer, change feature dimension to demanded
        self.refactor = nn.Linear(2048, feature_dim)


    def forward(self, x):
        feature = self.feature(x)
        global_feature = torch.flatten(feature, start_dim=1)
        global_feature = F.layer_norm(global_feature, [global_feature.size(-1)])
        feature = self.refactor(global_feature)
        return feature


class SimpleFeatureExtractor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.25),
            nn.Flatten(start_dim=1, end_dim=-1)
        )

        self.fc = nn.Sequential(
            nn.Linear(179776, 200),
            nn.ReLU(),
            nn.BatchNorm1d(200),
            nn.Dropout(0.25),
            nn.Linear(200, 100)
        )

        self.refactor = nn.Linear(100, feature_dim)


    def forward(self, x):
        feature = self.feature(x)
        #print(feature.shape)
        #print("check")
        feature = self.fc(feature)
        feature = self.refactor(feature)
        return feature


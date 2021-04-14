import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FeatureEmbedding(nn.Module):
    def __init__(self, model_name):
        super(FeatureEmbedding, self).__init__()
        
        self.model_name = model_name
        
        if self.model_name == 'resnet50':
            backbone = models.resnet50(pretrained = True)
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            num_features = backbone.fc.in_features
            self.avgpool = backbone.avgpool

        if self.model_name == 'densenet121':
            backbone = models.densenet121(pretrained = True)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            num_features = backbone.classifier.in_features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(num_features, 14)

    def forward(self, x):
        features = self.features(x)
        out = torch.relu(features)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out, features

class Attention1(nn.Module):
    def __init__(self):
        super(Attention1, self).__init__()
        self.conv1 = nn.Conv2d(2048, 2048, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(2048, 2048, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(2048, 14, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.relu(out)
        out = self.conv2(out)
        out = torch.relu(out)
        out = self.conv3(out)
        out = torch.sigmoid(out)
        return out

class ResidualAttention(nn.Module):
    def __init__(self):
        super(ResidualAttention, self).__init__()
        self.feature_embedding = FeatureEmbedding('resnet50')
        self.attention = Attention1()
        self.fc = nn.Linear(14 * 2048, 14)

    def forward(self, x):
        out, features = self.feature_embedding(x)
        score = self.attention(features)
        feature_weighted = self.discriminative_features(features, score)

        out = torch.flatten(feature_weighted, 1)
        out = self.fc(out)
        out = torch.sigmoid(out)
        return out

    def discriminative_features(self, feature, score):
        num_class = score.size(1)
        bz, num_channel, h, w = feature.size()

        feature_weighted = torch.zeros(bz, num_class, num_channel, dtype = score.dtype, device = score.device)

        for i in range(bz):
            feature_weighted[i] = (torch.matmul(score[i].unsqueeze(1) + 1, feature[i])).relu().mean(dim = (-2, -1))

        return feature_weighted
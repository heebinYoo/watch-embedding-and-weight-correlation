import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet50


class ConfidenceControlLoss(nn.Module):
    def __init__(self):
        super(ConfidenceControlLoss, self).__init__()

    def forward(self, pred, label):
        pred = F.softmax(pred, dim=1)
        loss = F.nll_loss(pred, label)

        return loss

# class ConfidenceControlLoss(nn.Module):
#     def __init__(self, in_features, out_features):
#         '''
#         Confidence Contol Softmax Loss
#         Three 'sample_type' available: ['high', 'low', 'aug']
#         임베딩 스페이스에서의 위치에 따라 컨피던스가 작은 애들이 low, augmented sample은 aug
#         '''
#         super(ConfidenceControlLoss, self).__init__()
#
#         self.loss = nn.CrossEntropyLoss()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#
#     def forward(self, x, labels, sample_type):
#
#         assert sample_type in ['high', 'low', 'aug', None]
#         if sample_type == 'high':
#             weight_from = int(0)
#             weight_to = int(self.weight.data.shape[0] / 2)
#         if sample_type == 'low':
#             weight_from = int(0)
#             weight_to = int(self.weight.data.shape[0])
#         if sample_type == 'aug':
#             weight_from = int(0)
#             weight_to = int(self.weight.data.shape[0])
#         if sample_type is None:
#             weight_from = int(0)
#             weight_to = int(self.weight.data.shape[0])
#
#         # x = self.convlayers(x)
#         # output = x.matmul(F.normalize(self.weight, dim=-1).t())
#         output = x.matmul(self.weight.t())
#         output = output[:, weight_from:weight_to]
#         cc_loss = self.loss(output, labels)
#
#         return cc_loss
#
#     def extra_repr(self):
#         return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)

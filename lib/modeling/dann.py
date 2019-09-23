'''
DANN module. See  https://arxiv.org/abs/1505.07818, https://arxiv.org/abs/1409.7495
'''

import torch
import torch.nn as nn
import torch.nn as nn


from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Dann_Head(nn.Module):
    def __init__(self, input_dim, domains_count, gamma, top_layers_count = 4):
        super(Dann_Head, self).__init__()
        self.progress = nn.Parameter(torch.Tensor([0,]), requires_grad = False) # must be updated from outside
        self.gamma = gamma
        self.map_size = 8
        self.top_layers_count = top_layers_count
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('dann_fc1',
                                          nn.Linear(top_layers_count * input_dim * self.map_size * self.map_size, 100))
        #self.domain_classifier.add_module('dann_bn1', nn.BatchNorm1d(100)) # - dows not work for batch = 1
        self.domain_classifier.add_module('dann_bn1', nn.GroupNorm(5, 100)) # - dows not work for batch = 1
        self.domain_classifier.add_module('dann_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dann_fc2', nn.Linear(100, domains_count))
        self.domain_classifier.add_module('dann_softmax', nn.LogSoftmax(dim=1))
        self.loss = nn.NLLLoss()

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, input_data, roidb):
        feature = torch.cat([nn.functional.adaptive_max_pool2d(data, self.map_size).flatten(start_dim=1)
                             for data in input_data[ -self.top_layers_count : ]], dim=1)
        alpha = 2. / (1. + torch.exp(-self.gamma * self.progress.data)) - 1
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)
        target = torch.Tensor([rec['dataset_id'] for rec in roidb]).to(domain_output.device)
        loss = self.loss(domain_output, target.long())
        return loss
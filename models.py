import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import torchvision
        import os
        
        os.environ['TORCH_HOME'] = 'cache' # hacky workaround to set model dir
        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Identity() # remote last fc
        self.fc = nn.Linear(2048, num_classes)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, x, return_h=False): # (bs, C, H, W)
        pooled_output = self.resnet50(x)
        logit = self.fc(pooled_output)
        if return_h:
            return logit, pooled_output
        else:
            return logit

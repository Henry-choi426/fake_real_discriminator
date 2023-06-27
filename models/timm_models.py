from torch import nn
import timm

class timm_model(nn.Module):
    def __init__(self, backbone, n_out, is_sigmoid, freeze_ratio = 0):
        super(timm_model, self).__init__()
        self.model = timm.create_model(model_name=backbone, pretrained=True, num_classes=n_out)
        self.is_sigmoid = is_sigmoid
        if freeze_ratio > 0:
            num_frozen_layers = int(freeze_ratio * len(list(self.model.parameters())))
            for i, param in enumerate(self.model.parameters()):
                if i < num_frozen_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        if self.is_sigmoid:
            x = nn.Sigmoid()(x)
        return x
    
        
from torch.nn import functional as F
import torch
import torch.nn as nn

def get_loss(loss_name: str, amp = None):
    
    if loss_name == 'crossentropy':
        return F.cross_entropy

    elif loss_name == 'bce':
        if amp:
            return nn.BCEWithLogitsLoss()
        else:
            return nn.BCELoss()

    else:
        print(f'{loss_name}: invalid loss name')
        return
    

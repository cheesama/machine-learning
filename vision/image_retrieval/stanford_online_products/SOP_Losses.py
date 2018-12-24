import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiBatch_Contrastive_Loss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, output, target):
        target = target.unsqueeze(1)

        similarity_matrix = torch.mm(output, output.transpose(1,0)).float()
        positive_pair_matrix = torch.eq(target, target.transpose(1,0)).float() - torch.eye(target.shape[0]).to(target.device.type).float()
        negative_pair_matrix = torch.ne(target, target.transpose(1,0)).float()
        
        positive_similarity = (1 - similarity_matrix) * positive_pair_matrix
        negative_similarity = (1 + similarity_matrix) * negative_pair_matrix

        return positive_similarity.sum() + negative_similarity.mean()


if __name__ == '__main__':
    criterion = MultiBatch_Contrastive_Loss()
    print (criterion._forward_pre_hooks)

        

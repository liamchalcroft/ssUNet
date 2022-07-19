from torch import nn
import torch

class BatchNormDimSwap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if len(x.shape)==3:
            x = x.permute(0,2,1)
        return x


class NOBS(nn.Module):
    """
    Adapted from implementation at https://github.com/Sherrylone/Align-Representation-with-Base
    """
    def __init__(self, groups=1):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        D, B = x.size()

        x = x.view(self.groups, D//self.groups, B)
        f_cov = (torch.bmm(x, x.transpose(1, 2)) / (B - 1)).float()
        out = torch.FloatTensor(device=f_cov.device)
        for i in range(f_cov.size(0)):
            f_cov = torch.where(torch.isnan(f_cov), torch.zeros_like(f_cov), f_cov)
            U, S, V = torch.svd(f_cov[i])
            diag = torch.diag(1.0 / torch.sqrt(S + 1e-5))
            rotate_mtx = torch.mm(torch.mm(U, diag), U.transpose(0, 1)).detach()
            x_out = torch.mm(rotate_mtx, x[i])
            out = torch.cat([out, x_out], dim=0)
        
        return out.detach()

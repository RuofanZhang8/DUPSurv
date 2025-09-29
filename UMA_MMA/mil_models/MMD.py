import torch

def rbf_kernel(x, y, gamma=None):
    """
    计算 RBF 核矩阵
    x: (B, N, F)
    y: (B, M, F)
    gamma: RBF 核参数
    return: (B, N, M) 的核矩阵
    """
    B, N, F = x.shape
    M = y.shape[1]
    
    # 计算欧式距离的平方
    x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)  # (B, N, 1)
    y_sq = torch.sum(y ** 2, dim=-1, keepdim=True)  # (B, M, 1)
    
    # 计算 pairwise 距离
    dist_sq = x_sq - 2 * torch.matmul(x, y.transpose(1, 2)) + y_sq.transpose(1, 2)  # (B, N, M)
    
    if gamma is None:
        gamma = 1.0 / (2.0 * torch.median(dist_sq))  # 自适应 gamma
    
    return torch.exp(-gamma * dist_sq) /2 # (B, N, M)

def batch_mmd(x, y, gamma=None):
    """
    计算 batch-wise MMD
    x: (B, N, F)  第一组数据
    y: (B, M, F)  第二组数据
    return: (B,) shape，每个 batch 的 MMD
    """
    K_xx = rbf_kernel(x, x, gamma).mean(dim=(1, 2))  # (B,)
    K_yy = rbf_kernel(y, y, gamma).mean(dim=(1, 2))  # (B,)
    K_xy = rbf_kernel(x, y, gamma).mean(dim=(1, 2))  # (B,)
    
    return K_xx + K_yy - 2 * K_xy


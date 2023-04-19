import torch
import torch.nn as nn

class SkeletonTransformer(nn.Module):
    def __init__(self, timesteps, kpts_dim, input_kpts_num, output_dim):
        '''
        Parameters:
            timesteps: Timesteps of input time-series data (equal to number of frames, mentioned as 'T' in the paper)
            kpts_dim: Dimentions of keypoints (usually 2 (x, y) or 3 (x, y, z))
            input_kpts_num: Number of joints in original keypoints (mentioned as 'N' in the paper)
            output_dim: Dimentions of output (mentioned as 'M' in the paper)
        Input:
            x: 2-dimentional tensor of shape (timesteps, input_kpts_num*kpts_dim)
        '''
        super().__init__()
        self.timesteps = timesteps
        self.kpts_dim = kpts_dim
        self.output_dim = output_dim

        # Weight of this module (mentioned as a matrix 'W' in the paper)
        self.W = nn.Parameter(nn.init.xavier_uniform_(torch.empty(input_kpts_num, output_dim)))

    def forward(self, x):
        x = torch.reshape(x, (self.timesteps, -1, self.kpts_dim))
        x = torch.permute(x, (0, 2, 1))
        x = torch.matmul(x, self.W)
        x = torch.permute(x, (0, 2, 1))
        x = torch.reshape(x, (self.timesteps, self.output_dim * self.kpts_dim))
        return x
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# N, C, H, W, K, kH, kW = (2, 4, 6, 6, 5, 3, 3)

configs = [
    (2, 4, 6, 6, 5, 3, 3),
    (4, 8, 32, 32, 4, 3, 3),
    (4, 8, 16, 16, 4, 3, 3),
    (4, 8, 16, 16, 4, 3, 3),
]


class wconv_3x3(nn.Module):
    """A ResNet block."""

    def __init__(self, weight_init):
        super(wconv_3x3, self).__init__()

        B_T = [1, 0, -1, 0, 0, 1, 1, 0, 0, -1, 1, 0, 0, 1, 0, -1]
        self.B_T = torch.from_numpy(np.array(B_T, dtype=np.float32).reshape(4, 4))

        G = [1, 0, 0, 1 / 2, 1 / 2, 1 / 2, 1 / 2, -1 / 2, 1 / 2, 0, 0, 1]
        G = torch.from_numpy(np.array(G, dtype=np.float32).reshape(4, 3))
        self.K = weight_init.size(0)
        self.U = nn.Parameter((G @ weight_init.detach() @ G.T).permute(2, 3, 0, 1))

        A_T = [1, 1, 1, 0, 0, 1, -1, -1]
        self.A_T = torch.from_numpy(np.array(A_T, dtype=np.float32).reshape(2, 4))

    def verify(self, conv_weight):
        input = torch.randint(
            high=100, low=-100, size=(4, conv_weight.size(1), 32, 32)
        ).float()
        output_ref = F.conv2d(input, conv_weight, padding=1).detach().numpy()
        output = self.forward(input).detach().numpy()
        np.testing.assert_allclose(output_ref, output)

    def forward(self, x):
        N, C, H, W = x.size()
        m = 2
        r = 3
        P = int(N * ((H + m - 1) // m) * ((W + m - 1) // m))
        a = m + r - 1

        tiled_x = F.unfold(x, kernel_size=(a, a), stride=m, padding=padding)
        tiled_x = (
            tiled_x.reshape(N, C, a, a, P // N)
            .permute(1, 0, 4, 2, 3)
            .reshape(C, P, a, a)
        )
        V = (self.B_T @ tiled_x @ self.B_T.T).permute(2, 3, 0, 1)

        M = self.U @ V
        Y = self.A_T @ M.permute(2, 3, 0, 1) @ self.A_T.T
        Y = (
            Y.reshape(self.K, N, (H + m - 1) // m, (W + m - 1) // m, m, m)
            .permute(1, 0, 2, 4, 3, 5)
            .reshape(N, self.K, H, W)
        )
        return Y


for config in configs:
    print(config)
    N, C, H, W, K, kH, kW = config
    padding = 1
    input = torch.randint(high=100, low=-100, size=(N, C, H, W)).float()
    weight = torch.randint(high=100, low=-100, size=(K, C, kH, kW)).float()
    output = F.conv2d(input, weight, padding=padding)

    wconv = wconv_3x3(weight_init=weight)
    Y = wconv(input)
    wconv.verify(weight)

    np.testing.assert_allclose(Y.detach().numpy(), output.numpy(), rtol=1e-5)

print("Success!")

import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 3, 1, 2],
                      [0, 2, 3, 4, 0],
                      [5, 0, 2, 1, 3],
                      [2, 0, 1, 3, 5],
                      [0, 2, 1, 4, 1]])
kernel = torch.tensor([[1, 2, 3],
                      [0, 1, 0],
                      [2, 1, 0]])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

output = F.conv2d(input, kernel, stride=1)
print(output)

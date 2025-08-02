import torch
import torch.nn.functional as F

input_tensor = torch.randn((10,)).cuda()
target_tensor = torch.randint(0, 5, (10,)).cuda()
print(input_tensor.size(), target_tensor.size())

loss = F.nll_loss(input_tensor, target_tensor)

print("Final Loss:", loss.item())

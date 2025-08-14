import torch

input_tensor = torch.rand(2, 10, 5, dtype=torch.float64)
output = torch.nn.InstanceNorm1d(num_features=10, track_running_stats=True)(input_tensor)

input_tensor = torch.rand(5, 10, 5, dtype=torch.float64)
output = torch.nn.InstanceNorm2d(num_features=5, track_running_stats=True)(input_tensor)

input_tensor = torch.rand(6, 10, 5, dtype=torch.float64)
output = torch.nn.InstanceNorm3d(num_features=5, track_running_stats=True)(input_tensor)

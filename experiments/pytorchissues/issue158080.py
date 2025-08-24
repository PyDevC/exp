import io

import torch

# build a state dict with many tensors sharing the same storage
n = 10000
buf = torch.empty(n)
state_dict = {f"w{i}": buf[i] for i in range(n)}

# save to a BytesIO buffer
buffer = io.BytesIO()
torch.save(state_dict, buffer)

# load from the BytesIO buffer
buffer.seek(0)
torch.load(buffer, map_location="cpu")
print("Loaded")
torch.load(buffer, map_location="meta")

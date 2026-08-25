import torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

dummy_inputs = torch.randint(0, 1000, (1, 128), dtype=torch.int32)
dummy_attention = torch.ones((1, 128), dtype=torch.int32)

with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    with record_function("model_inference"):
        out = model(dummy_inputs)

print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))

from typing import Callable

import torch
from transformers import AutoTokenizer, AutoModel

online_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
online_model = AutoModel.from_pretrained('bert-base-uncased')

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 6)

    def forward(self, x):
        return self.linear(x)

def bite_compile(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
    gm.graph.print_tabular()
    return gm

dummy_input_ids = torch.randint(0, 1000, (1, 128), dtype=torch.long)
dummy_attention_mask = torch.ones((1, 128), dtype=torch.long)

gm = torch.export.export(online_model, args=(dummy_input_ids,))
gm.graph.print_tabular()

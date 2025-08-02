#159460
import torch
import torch.nn as nn
from typing import Dict, Any

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.components: Dict[str, nn.Module] = {}
        self.parameters: Any = None  # This line triggers compile failure

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class ComponentModule(BaseModule):
    def add_component(self, name, component):
        self.components[name] = component

    def forward(self, *args, **kwargs):
        raise NotImplementedError

class ForwardPassModule(ComponentModule):
    def forward(self, x):
        for _, component in self.components.items():
            x = component(x)
        return x

class MyModel(ForwardPassModule):
    def __init__(self):
        super().__init__()
        self.add_component('layer1', nn.Linear(10, 20))
        self.add_component('relu', nn.ReLU())
        self.add_component('layer2', nn.Linear(20, 10))

def trigger_compile_failure():
    model = MyModel()
    x = torch.randn(1, 10)
    model(x)  # eager ok
    print("Eager model output pass ok!")
    compiled_model = torch.compile(model)
    print("Compiled model ok!")
    compiled_model(x)  # fails here
    print("Compiled model output pass ok!")

if __name__ == "__main__":
    trigger_compile_failure()

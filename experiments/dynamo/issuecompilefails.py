import torch
from collections import namedtuple
import dis

class FancyPair(namedtuple("FancyPair", ["a", "b"])):
    __slots__ = ()

    def __add__(self, other):
        return FancyPair(self.a + other.a, self.b + other.b)


class module(torch.nn.Module):
    def forward(self, x):
        base = FancyPair(x.sin(), x.cos())
        delta = FancyPair(x.relu(), x.tanh())
        combined = base + delta
        return combined.a * 0.5 - combined.b.exp()


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(3, requires_grad=True)
    m = module()
    output1 = m(x)
    print("eager:", output1)
    
    output2 = torch.compile(m)(x)
    print("compiled:", output2)
    dis.dis(torch.compile)

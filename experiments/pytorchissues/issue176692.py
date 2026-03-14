import torch

class LoggingTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, data):
        r = torch.Tensor._make_wrapper_subclass(
            cls, data.shape, dtype=data.dtype, device=data.device
        )
        r.elem = data
        return r

    def __repr__(self):
        return f"LoggingTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        unwrap = lambda t: t.elem if isinstance(t, LoggingTensor) else t
        wrap   = lambda t: LoggingTensor(t) if isinstance(t, torch.Tensor) else t

        out = func(
            *torch.utils._pytree.tree_map(unwrap, args),
            **torch.utils._pytree.tree_map(unwrap, kwargs),
        )
        return torch.utils._pytree.tree_map(wrap, out)

fn = lambda x, y: torch.mm(x, y)

a = LoggingTensor(torch.randn(3, 4))
b = LoggingTensor(torch.randn(4, 5))
breakpoint()

eager    = fn(a, b)
compiled = torch.compile(fn, backend="eager", fullgraph=False)(a, b)

print(f"eager   : {eager}")
print(f"compiled: {compiled}")
print(f"match   : {torch.allclose(eager, compiled)}")

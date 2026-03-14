import torch

def fn(x):
    grad_res, _ = torch.func.jacrev(lambda x: tuple([x]*2), has_aux=True)(x)
    return grad_res


compile_options = dict(
    backend="eager", fullgraph=True, dynamic=False
)

fn = torch.compile(fn, **compile_options)  # error goes away if we comment this line

vmapped_fn = torch.vmap(fn)
vmapped_compiled_fn = torch.compile(vmapped_fn, **compile_options)

xvals = torch.randn(128, dtype=torch.float64)

print()
print(f"{vmapped_fn(xvals).shape=}")
print(f"{vmapped_compiled_fn(xvals).shape=}")

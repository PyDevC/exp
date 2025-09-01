import torch
torch._inductor.config.force_disable_caches = True
# torch._inductor.config.triton.cooperative_reductions = True
@torch.compile(fullgraph=True, dynamic=False)
def _local_metrics(x):
    return x.abs().max(), x.abs().mean(), x.square().mean()
_local_metrics(torch.rand([512, 512, 512], device="cuda", requires_grad=False))

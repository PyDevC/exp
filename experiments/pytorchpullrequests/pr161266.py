import torch
from torch.func import vmap, jacrev, functional_call

class M(torch.nn.Module):
    def __init__(self, num_in, num_hidden):
        super(M,self).__init__()
        self.fc1 = torch.nn.Linear(num_in,num_hidden)
        self.fc2 = torch.nn.Linear(num_hidden,1)

    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# inputs
num_samples = 4096
num_input = 2
num_hidden=64

net = M(num_input, num_hidden)

x = torch.randn(num_samples, num_input)

y = net(x) #compute output (works fine)

#Compute trace of Hessian
def calc_hessian_trace(params, x):

  def output(params, x):
    return functional_call(net, params, x)

  _hessian = jacrev(jacrev(output, argnums=(1)), argnums=(1))(params, x)
  return _hessian.diagonal(0,-2,-1).sum(-1)

laplacian = vmap(calc_hessian_trace, in_dims=(None, 0))(dict(net.named_parameters()), x) #fail
platform linux -- Python 3.12.3, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/pydevc/pytorch
configfile: pytest.ini
plugins: hypothesis-6.138.3
collected 0 items / 1 error
Running 0 items in this shard

=========================================================================================== ERRORS ============================================================================================
__________________________________________________________________ ERROR collecting test/inductor/test_compiled_autograd.py ___________________________________________________________________
ImportError while importing test module '/home/pydevc/pytorch/test/inductor/test_compiled_autograd.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/usr/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
test/inductor/test_compiled_autograd.py:47: in <module>
    from torch.testing._internal.hop_db import hop_db
torch/testing/_internal/hop_db.py:7: in <module>
    from functorch.experimental.control_flow import map
functorch/experimental/__init__.py:2: in <module>
    from functorch import functionalize
E   ImportError: cannot import name 'functionalize' from 'functorch' (unknown location)
=================================================================================== short test summary info ===================================================================================
ERROR test/inductor/test_compiled_autograd.py
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
====================================================================================== 1 error in 3.17s =======================================================================================

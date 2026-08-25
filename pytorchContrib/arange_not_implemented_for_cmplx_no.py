import torch

print(f"{torch.arange(3, dtype=torch.complex64)=}")
print(f"{torch.arange(1, 7, step=4, dtype=torch.complex128)=}")
# print(f"{torch.arange(1, 10+1j, dtype=torch.complex64)=}") // not supported
print(f"{torch.arange(1, 10, step=2, dtype=torch.complex64)=}")
# print(f"{torch.arange(1+4j, 10, step=2, dtype=torch.complex64)=}") // not supported
# print(f"{torch.arange(1, 10+1j, step=2, dtype=torch.complex64)=}") // not supported
# print(f"{torch.arange(1, 10, step=2+1j, dtype=torch.complex64)=}")// not supported
# print(f"{torch.arange(1+1j, 10, step=2+0.2j, dtype=torch.complex64)=}") // not supported
# print(f"{torch.arange(1, 1+3j, step=1+1j, dtype=torch.complex64)=}") // not supported
# print(f"{torch.arange(1+0j, 1+3j, step=1+1j, dtype=torch.complex64)=}") // not supported

# The reason why these are not supported is because we are right now converting them into doubles which is kind of wrong and we need to know how does CPU handles Complex Numbers if there is some specific way then we will write about it sure soon which is kind of wrong and we need to know how does CPU handles Complex Numbers if there is some specific way then we will write about it sure soon.

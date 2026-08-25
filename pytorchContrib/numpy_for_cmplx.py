import numpy as np

print(f"{np.arange(3, dtype=np.complex64)=}")
print(f"{np.arange(3+0j, dtype=np.complex64)=}")
print(f"{np.arange(1, 3, dtype=np.complex64)=}")
print(f"{np.arange(1, 10+1j, dtype=np.complex64)=}")
print(f"{np.arange(1, 10, step=2, dtype=np.complex64)=}")
print(f"{np.arange(1+4j, 10, step=2, dtype=np.complex64)=}")
print(f"{np.arange(1, 10+1j, step=2, dtype=np.complex64)=}")
print(f"{np.arange(1, 10, step=2+1j, dtype=np.complex64)=}")
print(f"{np.arange(1+1j, 10, step=2+0.2j, dtype=np.complex64)=}")
print(f"{np.arange(1, 1+3j, step=1+1j, dtype=np.complex64)=}")
print(f"{np.arange(1+0j, 1+3j, step=1+1j, dtype=np.complex64)=}")

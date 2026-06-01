# Developer System Environment (AGENTS.md)

This document provides details on the hardware, software, and execution configurations for running the Gomoku training codebase on the local developer machine.

## Hardware & System Details

- **GPU**: **Intel Arc B580** (Battlemage Xe2 architecture)
  - VRAM: 12 GB GDDR6
  - API Support: DirectX 12 Ultimate, Vulkan, OpenCL, and Intel oneAPI / SYCL.
- **CPU**: Intel Core processor
- **OS**: Windows 11

## Software Environment

- **Python Version**: `3.12.x`
- **Deep Learning Framework**: **PyTorch 2.12.0+xpu**
  - Includes custom Intel XPU backend integrations (Intel Extension for PyTorch / IPEX) allowing native accelerated execution on Intel Arc graphics hardware.
- **Triton Compiler**: `triton-xpu==3.7.1`

---

## PyTorch Device Target Configuration (XPU)

On Intel Arc graphics platforms, PyTorch accesses GPU acceleration using the `xpu` device backend instead of the standard NVIDIA `cuda` backend.

### 1. Verification of Intel GPU in Python
You can verify GPU accessibility and oneAPI configuration in your Python shell:
```python
import torch

print("PyTorch Version:", torch.__version__)
print("XPU Available  :", hasattr(torch, "xpu") and torch.xpu.is_available())
if hasattr(torch, "xpu") and torch.xpu.is_available():
    print("GPU Device Name:", torch.xpu.get_device_name(0))
```

### 2. Device Selection Heuristics in Code
The codebase implements a unified hardware fallback path in `src/trainer.py` to support NVIDIA, Intel Arc, and CPU runtimes dynamically:
```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")
```

### 3. Pinning Memory
For PyTorch `DataLoader` operations, pinning memory speeds up transfers between host and Intel GPU. If utilizing XPU, verify that `pin_memory=True` is enabled in `DataLoader` instances.

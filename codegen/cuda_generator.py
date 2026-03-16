"""
CUDA code generation (prototype).

Later steps will:
  - emit CUDA .cu code for kernels (relu/maxpool/conv)
  - compile via NVRTC (preferred) or a build step with nvcc
  - load and launch from Python runtime
"""


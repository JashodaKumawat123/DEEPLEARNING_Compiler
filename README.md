## Deep Learning Kernel Optimization Compiler

A research-style mini ML compiler prototype that:

- takes a small PyTorch model
- extracts a computation graph
- runs basic optimization passes (fusion, constant folding, memory planning)
- generates CUDA kernels (prototype)
- executes and benchmarks vs baselines

This repo is designed to run on **Linux with CUDA installed**.

### Quick start (graph extraction demo)

Create a Python env with PyTorch installed (CUDA-enabled build recommended), then run:

```bash
python main.py --print-graph
```


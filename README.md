## Deep Learning Kernel Optimization Compiler

A prototype machine learning compiler that converts neural network computation graphs into optimized CUDA kernels for GPU execution.
The system performs graph-level compiler optimizations such as operator fusion and automatically generates GPU kernels to improve inference performance.

This project demonstrates concepts used in modern ML compiler systems such as TensorRT, Apache TVM, and XLA.

### Overview

Deep learning frameworks typically execute neural networks as a sequence of independent GPU kernels. This can introduce overhead due to:

- repeated memory transfers
- excessive kernel launches
- inefficient scheduling

This project builds a mini deep learning compiler that analyzes the neural network graph and performs optimizations before execution.

The compiler pipeline performs:

- Model loading
- Graph extraction
- Graph optimization passes
- CUDA kernel generation
- GPU execution
- Performance benchmarking

### Architecture

```text
Neural Network Model (PyTorch)
                │
                ▼
        Graph Extraction
                │
                ▼
  Graph Intermediate Representation
                │
                ▼
       Optimization Pass Engine
     (Operator Fusion, Memory Opt)
                │
                ▼
        CUDA Code Generation
                │
                ▼
             GPU Runtime
                │
                ▼
        Performance Benchmark
```

### Project Structure

```text
deep-learning-kernel-compiler
│
├── models
│   └── sample_model.py
│
├── graph
│   ├── graph_ir.py
│   └── graph_extractor.py
│
├── optimizer
│   ├── operator_fusion.py
│   └── constant_folding.py
│
├── codegen
│   └── cuda_generator.py
│
├── runtime
│   └── executor.py
│
├── benchmarks
│   └── benchmark.py
│
├── kernels
│   └── conv_relu_kernel.cu
│
└── main.py
```

### Key Features

#### Graph Intermediate Representation

The compiler converts neural networks into a `GraphIR` structure.

Example graph:

```text
Input
 ↓
Conv2D
 ↓
ReLU
 ↓
MaxPool
 ↓
Dense
```

Each operation is represented as a node containing:

- operation type
- inputs
- outputs

This representation allows the compiler to analyze and transform the network.

#### Optimization Passes

The system performs compiler optimizations on the computational graph.

**Operator Fusion**

The compiler detects patterns such as:

```text
Conv2D → ReLU
```

and fuses them into:

```text
Conv2DReLU
```

Benefits:

- fewer kernel launches
- reduced GPU memory transfers
- improved inference latency

This optimization is commonly used in ML compilers such as TensorRT.

### CUDA Kernel Generation

After optimization, the compiler generates CUDA kernels automatically.

Example fused kernel:

```cpp
__global__ void conv_relu_kernel(float* input,
                                 float* weight,
                                 float* output,
                                 int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        float val = input[idx] * weight[idx];
        output[idx] = val > 0 ? val : 0;
    }
}
```

This kernel performs convolution and ReLU activation in a single GPU kernel.

### GPU Runtime

The runtime engine performs:

- CUDA kernel compilation
- GPU memory allocation
- kernel launch scheduling
- execution synchronization

This enables optimized GPU inference.

### Performance Benchmarking

The project includes a benchmarking module that compares:

| Execution Mode   | Description                      |
|------------------|----------------------------------|
| CPU              | Standard model execution         |
| GPU (Naive)      | GPU execution without optimization |
| GPU (Optimized)  | Execution using compiler optimizations |

Example output:

```text
CPU Execution Time: 120 ms
Naive GPU Execution: 30 ms
Optimized GPU Execution: 10 ms

Speedup: 3x
```

### Installation

**Requirements**

- Python 3.9+
- CUDA Toolkit
- PyTorch
- Linux environment

Install dependencies:

```bash
pip install torch numpy
```

Ensure CUDA is installed:

```bash
nvcc --version
```

### Running the Project

Run the full compiler pipeline:

```bash
python main.py
```

The system will:

- Load the neural network model
- Extract the computation graph
- Apply optimization passes
- Generate CUDA kernels
- Execute GPU kernels
- Display performance benchmarks

### Example Optimization

Before optimization:

```text
Input → Conv2D → ReLU → MaxPool
```

After optimization:

```text
Input → Conv2DReLU → MaxPool
```

This reduces GPU kernel launches and improves performance.

### Technologies Used

**Languages**

- Python
- C++
- CUDA

**Libraries**

- PyTorch

**Concepts**

- compiler optimization
- graph intermediate representation
- CUDA kernel programming
- GPU performance optimization

### Future Improvements

Planned extensions include:

- automatic kernel tuning
- memory reuse optimization
- tensor scheduling
- LLVM-based optimization passes
- hardware-aware scheduling

These features would make the compiler closer to production systems like Apache TVM.

### Learning Outcomes

This project demonstrates:

- ML compiler architecture
- graph optimization techniques
- GPU kernel generation
- CUDA programming
- deep learning performance optimization


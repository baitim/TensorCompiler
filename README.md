<h1 align="center">TensorCompiler</h1>

## Description

Implementation of a Tensor Compiler with ONNX frontend and MLIR/LLVM backend.  
Parses ONNX models, builds a computational graph, and can emit MLIR, LLVM IR, or native assembly for a specified target architecture.

## How to integrate
 
Use [storage](https://github.com/baitim/TensorCompiler), project = "tensor_compiler", version = "1.0", user = "baitim"

## How to run

1. Clone <br>
    <code>git clone https://github.com/baitim/TensorCompiler.git</code>

2. Go to folder <br>
    <code>cd TensorCompiler</code>

3. Prepare conan <br>
    <code>uv sync --group dev; source .venv/bin/activate</code><br>
    <code>conan profile detect --force</code>

4. Init dependencies <br>
    <code>conan install . --build=missing -s build_type=Release</code>

5. Build <br>
    <code>cmake --preset release; cmake --build build/Release</code>

6. Run <br>
    <code>./build/Release/TensorCompiler/tensor-compiler &lt;model.onnx&gt; [options]</code>

## Supported options

| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--graphviz-dump` | Generate GraphViz DOT file of the graph |
| `--print-graph` | Print detailed graph information (tensors, nodes, attributes) |
| `--emit-mlir` | Emit MLIR textual representation |
| `--emit-llvm` | Emit LLVM IR (requires lowering from MLIR) |
| `--emit-asm` | Emit native assembly for the target architecture |
| `--target=<triple>` | Target triple (e.g., `x86_64-unknown-linux-gnu`) |
| `--opt-level=<0-3>` | Optimization level (default: 2) |
| `--output=<filename>` | Base name for output files (without extension) |

### Examples

```bash
# Parse and display graph info
./build/Release/TensorCompiler/tensor-compiler model.onnx --print-graph

# Emit MLIR to output.mlir
./build/Release/TensorCompiler/tensor-compiler model.onnx --emit-mlir --output=model

# Generate assembly for ARM64
./build/Release/TensorCompiler/tensor-compiler model.onnx --emit-asm --target=aarch64-unknown-linux-gnu

## How to test

* Run testing <br>
    <code>ctest --test-dir build/Release --output-on-failure</code>

<p align="center"><img src="https://github.com/baitim/TensorCompiler/blob/main/images/cat.gif" width="50%"></p>

## Support
**This project is created by [baitim](https://t.me/bai_tim)**
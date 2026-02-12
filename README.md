<h1 align="center">TensorCompiler</h1>

## Description

 Implementation of the Tensor Compiler Frontend<br>

## How to integrate
 
 use [storage](https://github.com/baitim/TensorCompiler), project = "tensor_compiler", version = "1.0", user = "baitim"

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
    <code>./build/Release/TensorCompiler/tensor_compiler \<program\></code>

## Supported options

* print options info <code>--help</code>
* generate GraphViz dot file <code>--graphviz-dump</code>
* print graph info <code>--print-graph</code>

<p align="center"><img src="https://github.com/baitim/TensorCompiler/blob/main/images/cat.gif" width="50%"></p>

## Support
**This project is created by [baitim](https://t.me/bai_tim)**
import argparse
import os
import subprocess
import sys
import numpy as np

def run_onnxruntime(model_path, input_data):
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    result = sess.run(None, {input_name: input_data})
    return result[0]

def run_compiler_executor(compiler_bin, model_path, input_bin_path):
    cmd = [compiler_bin, model_path, "--execute", f"--input-data={input_bin_path}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Compiler failed:\n{result.stderr}")

    output_line = result.stdout.strip()
    for line in output_line.splitlines():
        if "[" in line and "]" in line:
            content = line[line.index("[") + 1 : line.rindex("]")]
            values = [float(x.strip()) for x in content.split(",")]
            return np.array(values, dtype=np.float32)
    raise RuntimeError(f"Could not parse compiler output:\n{result.stdout}")

def main():
    parser = argparse.ArgumentParser(description="Compare tensor compiler output with onnxruntime")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--input", required=True, help="Path to binary float32 input file")
    parser.add_argument("--compiler", required=True, help="Path to tensor-compiler binary")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    input_data = np.fromfile(args.input, dtype=np.float32)

    import onnx
    model = onnx.load(args.model)
    input_shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    input_data = input_data.reshape(input_shape)

    print("Running onnxruntime...")
    ref_output = run_onnxruntime(args.model, input_data)
    print(f"  Reference output shape: {ref_output.shape}")
    print(f"  Reference output[:5]: {ref_output.flatten()[:5]}")

    print("Running tensor compiler executor...")
    cmp_output = run_compiler_executor(args.compiler, args.model, args.input)
    print(f"  Compiler output shape: {cmp_output.shape}")
    print(f"  Compiler output[:5]: {cmp_output.flatten()[:5]}")

    ref_flat = ref_output.flatten()
    cmp_flat = cmp_output.flatten()

    if np.allclose(ref_flat, cmp_flat, rtol=args.rtol, atol=args.atol):
        print("\nOUTPUTS MATCH (within tolerance)")
        max_diff = np.max(np.abs(ref_flat - cmp_flat))
        print(f"  Max absolute difference: {max_diff:.2e}")
        return 0
    else:
        print("\nOUTPUTS DO NOT MATCH")
        max_diff = np.max(np.abs(ref_flat - cmp_flat))
        print(f"  Max absolute difference: {max_diff:.2e}")
        mismatches = np.where(~np.isclose(ref_flat, cmp_flat, rtol=args.rtol, atol=args.atol))[0]
        print(f"  Mismatching indices: {mismatches[:10]}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

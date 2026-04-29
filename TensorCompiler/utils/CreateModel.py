import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np
import os

np.random.seed(42)

def make_tensor(name, shape, dtype=TensorProto.FLOAT, scale=0.1):
    data = (np.random.randn(*shape) * scale).astype(np.float32)
    return numpy_helper.from_array(data, name=name)

def build_simple_convnet():
    conv_w = make_tensor("conv_w", [6, 1, 5, 5])

    reshape_shape = numpy_helper.from_array(np.array([1, 3456], dtype=np.int64), name="reshape_shape")

    fc1_w = make_tensor("fc1_w", [3456, 120], scale=0.01)
    fc1_b = make_tensor("fc1_b", [120], scale=0.01)

    fc2_w = make_tensor("fc2_w", [120, 10], scale=0.01)
    fc2_b = make_tensor("fc2_b", [10], scale=0.01)

    nodes = [
        helper.make_node("Conv",   ["input", "conv_w"],             ["conv_out"],    kernel_shape=[5, 5]),
        helper.make_node("Relu",   ["conv_out"],                    ["relu1_out"]),
        helper.make_node("Reshape", ["relu1_out", "reshape_shape"], ["reshape_out"]),
        helper.make_node("Gemm",   ["reshape_out", "fc1_w", "fc1_b"], ["gemm1_out"]),
        helper.make_node("Relu",   ["gemm1_out"],                   ["relu2_out"]),
        helper.make_node("Gemm",   ["relu2_out", "fc2_w", "fc2_b"], ["output"]),
    ]

    graph = helper.make_graph(
        nodes,
        "SimpleConvNet",
        [helper.make_tensor_value_info("input",  TensorProto.FLOAT, [1, 1, 28, 28])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
        initializer=[conv_w, reshape_shape, fc1_w, fc1_b, fc2_w, fc2_b],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    return model


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "test")
    os.makedirs(out_dir, exist_ok=True)

    model = build_simple_convnet()
    path = os.path.join(out_dir, "simple_convnet.onnx")
    onnx.save(model, path)
    print(f"Model saved to {path}")

    input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
    input_path = os.path.join(out_dir, "simple_convnet_input.bin")
    input_data.tofile(input_path)
    print(f"Input data saved to {input_path}")

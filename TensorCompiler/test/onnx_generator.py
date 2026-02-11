import onnx
from onnx import helper, TensorProto
import numpy as np
import random
import os

N = 4
ops_list = ['Add', 'Mul', 'Conv', 'Relu', 'MatMul', 'Gemm']

if N >= len(ops_list):
    chosen_ops = ops_list.copy()
    chosen_ops.extend(random.choices(ops_list, k=N-6))
    random.shuffle(chosen_ops)
else:
    chosen_ops = random.sample(ops_list, N)
    random.shuffle(chosen_ops)

input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
nodes, initializers = [], []
current = "input"

for idx, op in enumerate(chosen_ops):
    out_name = f"{op.lower()}_out_{idx}"
    
    if op == 'Conv':
        shape_4d = helper.make_tensor(f"shape_4d_{idx}", TensorProto.INT64, [4], np.array([1,4,1,1], dtype=np.int64))
        shape_2d = helper.make_tensor(f"shape_2d_{idx}", TensorProto.INT64, [2], np.array([1,4], dtype=np.int64))
        conv_w = helper.make_tensor(f"conv_w_{idx}", TensorProto.FLOAT, [4,4,1,1], np.ones((4,4,1,1), dtype=np.float32))
        conv_b = helper.make_tensor(f"conv_b_{idx}", TensorProto.FLOAT, [4], np.zeros(4, dtype=np.float32))
        initializers.extend([shape_4d, shape_2d, conv_w, conv_b])
        
        nodes.append(helper.make_node("Reshape", [current, f"shape_4d_{idx}"], [f"reshape1_{idx}"]))
        nodes.append(helper.make_node("Conv", [f"reshape1_{idx}", f"conv_w_{idx}", f"conv_b_{idx}"], [f"conv_{idx}"], kernel_shape=[1,1]))
        nodes.append(helper.make_node("Reshape", [f"conv_{idx}", f"shape_2d_{idx}"], [out_name]))
    
    elif op == 'Relu':
        nodes.append(helper.make_node("Relu", [current], [out_name]))
    
    elif op in ['Add', 'Mul']:
        const = helper.make_tensor(f"{op.lower()}_const_{idx}", TensorProto.FLOAT, [1,4], np.ones((1,4), dtype=np.float32))
        initializers.append(const)
        nodes.append(helper.make_node(op, [current, f"{op.lower()}_const_{idx}"], [out_name]))
    
    elif op == 'MatMul':
        w = helper.make_tensor(f"matmul_w_{idx}", TensorProto.FLOAT, [4,4], np.ones((4,4), dtype=np.float32))
        initializers.append(w)
        nodes.append(helper.make_node("MatMul", [current, f"matmul_w_{idx}"], [out_name]))
    
    elif op == 'Gemm':
        w = helper.make_tensor(f"gemm_w_{idx}", TensorProto.FLOAT, [4,4], np.ones((4,4), dtype=np.float32))
        b = helper.make_tensor(f"gemm_b_{idx}", TensorProto.FLOAT, [4], np.zeros(4, dtype=np.float32))
        initializers.extend([w, b])
        nodes.append(helper.make_node("Gemm", [current, f"gemm_w_{idx}", f"gemm_b_{idx}"], [out_name]))
    
    current = out_name

output_tensor = helper.make_tensor_value_info(current, TensorProto.FLOAT, [1,4])
graph = helper.make_graph(nodes, "RandomOpsGraph", [input_tensor], [output_tensor], initializer=initializers)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

output_path = os.path.join(os.path.dirname(__file__), "random_ops.onnx")
onnx.save(model, output_path)
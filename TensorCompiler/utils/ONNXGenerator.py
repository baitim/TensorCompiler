import onnx
from onnx import helper, TensorProto
import numpy as np
import random
import os

NUM_TESTS = 10
OPS_PER_TEST = 15
OPS_LIST = ['Add', 'Mul', 'Conv', 'Relu', 'MatMul', 'Gemm']

def dtype_to_string(data_type):
    return {
        TensorProto.FLOAT: "FLOAT",
        TensorProto.INT32: "INT32",
        TensorProto.INT64: "INT64",
        TensorProto.BOOL: "BOOL",
        TensorProto.STRING: "STRING",
        TensorProto.UINT8: "UINT8",
        TensorProto.DOUBLE: "DOUBLE",
    }.get(data_type, "UNKNOWN")

def format_shape(shape):
    return '[' + ', '.join(str(d) for d in shape) + ']'

def generate_expected(graph_name, nodes, initializers, input_name, output_name):
    input_shape = [1, 4]
    output_shape = [1, 4]

    all_tensors = {input_name}
    for init in initializers:
        all_tensors.add(init.name)
    for node in nodes:
        all_tensors.update(node.output)
    tensor_count = len(all_tensors)

    lines = []
    lines.append('=== Computational Graph Summary ===')
    lines.append(f'Name: {graph_name}')
    lines.append(f'Nodes: {len(nodes)}')
    lines.append(f'Tensors: {tensor_count}')
    lines.append(f'Input tensors: 1')
    lines.append(f'Output tensors: 1')
    lines.append(f'Initializers: {len(initializers)}')
    lines.append('')
    lines.append('=== Detailed Information ===')
    lines.append('')
    lines.append('Input Tensors:')
    lines.append(f'  - {input_name} {format_shape(input_shape)}')
    lines.append('')
    lines.append('Output Tensors:')
    lines.append(f'  - {output_name} {format_shape(output_shape)}')
    lines.append('')
    lines.append('Initializers (weights):')
    for init in initializers:
        shape_str = format_shape(list(init.dims))
        dtype_str = dtype_to_string(init.data_type)
        lines.append(f'  - {init.name} {shape_str} dtype={dtype_str}')
    lines.append('')
    lines.append('Nodes:')
    for node in nodes:
        lines.append(f'  - {node.name} [{node.op_type}]')
        inputs_str = ' '.join(node.input)
        lines.append(f'    Inputs: {inputs_str} ')
        outputs_str = ' '.join(node.output)
        lines.append(f'    Outputs: {outputs_str} ')
        attrs = []
        for attr in node.attribute:
            if attr.name == 'kernel_shape':
                attrs.append(('kernel_shape', 'INTS', list(attr.ints)))
        lines.append(f'    Attributes ({len(attrs)}):')
        for name, typ, val in attrs:
            if typ == 'INTS':
                val_str = '[' + ', '.join(str(v) for v in val) + ']'
            else:
                val_str = str(val)
            lines.append(f'      {name} ({typ}): {val_str}')
    return '\n'.join(lines) + '\n'

def create_conv_ops(current, pos, out_name):
    """Create Conv operation with necessary reshape tensors and nodes"""
    nodes = []
    initializers = []
    
    shape_4d = helper.make_tensor(
        f'shape_4d_{pos}', TensorProto.INT64, [4],
        np.array([1,4,1,1], dtype=np.int64)
    )
    shape_2d = helper.make_tensor(
        f'shape_2d_{pos}', TensorProto.INT64, [2],
        np.array([1,4], dtype=np.int64)
    )
    conv_w = helper.make_tensor(
        f'conv_w_{pos}', TensorProto.FLOAT, [4,4,1,1],
        np.ones((4,4,1,1), dtype=np.float32)
    )
    conv_b = helper.make_tensor(
        f'conv_b_{pos}', TensorProto.FLOAT, [4],
        np.zeros(4, dtype=np.float32)
    )
    initializers.extend([shape_4d, shape_2d, conv_w, conv_b])

    nodes.append(helper.make_node(
        'Reshape', [current, f'shape_4d_{pos}'],
        [f'reshape1_{pos}'], name=f'reshape1_{pos}'
    ))
    nodes.append(helper.make_node(
        'Conv', [f'reshape1_{pos}', f'conv_w_{pos}', f'conv_b_{pos}'],
        [f'conv_{pos}'], kernel_shape=[1,1], name=f'conv_{pos}'
    ))
    nodes.append(helper.make_node(
        'Reshape', [f'conv_{pos}', f'shape_2d_{pos}'],
        [out_name], name=f'reshape2_{pos}'
    ))
    
    return nodes, initializers

def create_relu_op(current, pos, out_name):
    """Create ReLU operation"""
    nodes = []
    nodes.append(helper.make_node(
        'Relu', [current], [out_name], name=f'Relu_{pos}'
    ))
    return nodes

def create_binary_op(op_type, current, pos, out_name):
    """Create Add or Mul operation with constant tensor"""
    nodes = []
    initializers = []
    
    const = helper.make_tensor(
        f'{op_type.lower()}_const_{pos}', TensorProto.FLOAT, [1,4],
        np.ones((1,4), dtype=np.float32)
    )
    initializers.append(const)
    nodes.append(helper.make_node(
        op_type, [current, f'{op_type.lower()}_const_{pos}'],
        [out_name], name=f'{op_type}_{pos}'
    ))
    
    return nodes, initializers

def create_matmul_op(current, pos, out_name):
    """Create MatMul operation with weight tensor"""
    nodes = []
    initializers = []
    
    w = helper.make_tensor(
        f'matmul_w_{pos}', TensorProto.FLOAT, [4,4],
        np.ones((4,4), dtype=np.float32)
    )
    initializers.append(w)
    nodes.append(helper.make_node(
        'MatMul', [current, f'matmul_w_{pos}'],
        [out_name], name=f'MatMul_{pos}'
    ))
    
    return nodes, initializers

def create_gemm_op(current, pos, out_name):
    """Create Gemm operation with weight and bias tensors"""
    nodes = []
    initializers = []
    
    w = helper.make_tensor(
        f'gemm_w_{pos}', TensorProto.FLOAT, [4,4],
        np.ones((4,4), dtype=np.float32)
    )
    b = helper.make_tensor(
        f'gemm_b_{pos}', TensorProto.FLOAT, [4],
        np.zeros(4, dtype=np.float32)
    )
    initializers.extend([w, b])
    nodes.append(helper.make_node(
        'Gemm', [current, f'gemm_w_{pos}', f'gemm_b_{pos}'],
        [out_name], name=f'Gemm_{pos}'
    ))
    
    return nodes, initializers

def create_operation(op_type, current, pos, out_name):
    """Factory function to create operations based on op_type"""
    if op_type == 'Conv':
        return create_conv_ops(current, pos, out_name)
    elif op_type == 'Relu':
        return create_relu_op(current, pos, out_name), []
    elif op_type in ['Add', 'Mul']:
        return create_binary_op(op_type, current, pos, out_name)
    elif op_type == 'MatMul':
        return create_matmul_op(current, pos, out_name)
    elif op_type == 'Gemm':
        return create_gemm_op(current, pos, out_name)
    else:
        return [], []

def main():
    random.seed(42)
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'test', 'end_to_end')
    os.makedirs(out_dir, exist_ok=True)

    for test_idx in range(NUM_TESTS):
        chosen_ops = [random.choice(OPS_LIST) for _ in range(OPS_PER_TEST)]
        
        all_nodes = []
        all_initializers = []
        current = 'input'

        for pos, op in enumerate(chosen_ops):
            out_name = f'{op.lower()}_out_{pos}'
            
            if op in ['Conv', 'Add', 'Mul', 'MatMul', 'Gemm']:
                nodes, initializers = create_operation(op, current, pos, out_name)
                all_nodes.extend(nodes)
                all_initializers.extend(initializers)
            elif op == 'Relu':
                nodes = create_operation(op, current, pos, out_name)[0]
                all_nodes.extend(nodes)
            
            current = out_name

        graph_name = f'RandomOpsGraph_{test_idx}'
        input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1,4])
        output_tensor = helper.make_tensor_value_info(current, TensorProto.FLOAT, [1,4])
        
        graph = helper.make_graph(
            all_nodes,
            graph_name,
            [input_tensor],
            [output_tensor],
            initializer=all_initializers
        )
        
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
        expected_output = generate_expected(graph_name, all_nodes, all_initializers, 'input', current)

        metadata = model.metadata_props.add()
        metadata.key = "expected_output"
        metadata.value = expected_output

        onnx_path = os.path.join(out_dir, f'test_{test_idx}.onnx')
        onnx.save(model, onnx_path)
        print(f'Generated {onnx_path}')

if __name__ == '__main__':
    main()
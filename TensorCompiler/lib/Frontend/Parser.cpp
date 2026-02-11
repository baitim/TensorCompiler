#include "Parser.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <onnx/onnx_pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

DataType ONNXParser::convert_onnx_type(int32_t onnx_type) {
    switch(onnx_type) {
        case onnx::TensorProto_DataType_FLOAT: return DataType::FLOAT;
        case onnx::TensorProto_DataType_INT32: return DataType::INT32;
        case onnx::TensorProto_DataType_INT64: return DataType::INT64;
        case onnx::TensorProto_DataType_BOOL: return DataType::BOOL;
        case onnx::TensorProto_DataType_STRING: return DataType::STRING;
        case onnx::TensorProto_DataType_UINT8: return DataType::UINT8;
        case onnx::TensorProto_DataType_DOUBLE: return DataType::DOUBLE;
        default: return DataType::UNKNOWN;
    }
}

OpType ONNXParser::convert_op_type(const std::string& op_type_str) {
    if (op_type_str == "Add") return OpType::ADD;
    if (op_type_str == "Mul") return OpType::MUL;
    if (op_type_str == "Conv") return OpType::CONV;
    if (op_type_str == "Relu") return OpType::RELU;
    if (op_type_str == "MatMul") return OpType::MATMUL;
    if (op_type_str == "Gemm") return OpType::GEMM;
    if (op_type_str == "Reshape") return OpType::RESHAPE;
    return OpType::UNKNOWN;
}

void ONNXParser::parse_initializers(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph) {
    for (int i = 0; i < onnx_graph.initializer_size(); ++i) {
        const onnx::TensorProto& tensor_proto = onnx_graph.initializer(i);

        std::string tensor_name = tensor_proto.name();

        std::vector<int64_t> shape;
        shape.reserve(tensor_proto.dims_size());
        for (int j = 0; j < tensor_proto.dims_size(); ++j)
            shape.push_back(tensor_proto.dims(j));

        DataType dtype = convert_onnx_type(tensor_proto.data_type());

        Tensor* tensor = graph.create_tensor_with_data(tensor_name, shape, dtype);
        tensor->is_initializer = true;
        graph.initializers.push_back(tensor);
    }
}

void ONNXParser::parse_inputs(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph) {
    for (int i = 0; i < onnx_graph.input_size(); ++i) {
        const onnx::ValueInfoProto& value_info = onnx_graph.input(i);
        std::string tensor_name = value_info.name();

        Tensor* tensor = graph.get_tensor(tensor_name);
        if (!tensor) {
            tensor = graph.create_tensor(tensor_name);

            if (value_info.has_type() && value_info.type().has_tensor_type()) {
                const auto& tensor_type = value_info.type().tensor_type();
                DataType dtype = convert_onnx_type(tensor_type.elem_type());
                tensor->dtype = dtype;

                if (tensor_type.has_shape()) {
                    const auto& shape_proto = tensor_type.shape();
                    tensor->shape.reserve(shape_proto.dim_size());
                    for (int j = 0; j < shape_proto.dim_size(); ++j) {
                        const auto& dim = shape_proto.dim(j);
                        if (dim.has_dim_value()) {
                            tensor->shape.push_back(dim.dim_value());
                        } else if (dim.has_dim_param()) {
                            tensor->shape.push_back(-1);
                        }
                    }
                }
            }
        }

        graph.input_tensors.push_back(tensor);
    }
}

void ONNXParser::parse_outputs(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph) {
    for (int i = 0; i < onnx_graph.output_size(); ++i) {
        const onnx::ValueInfoProto& value_info = onnx_graph.output(i);
        std::string tensor_name = value_info.name();
        Tensor* tensor = graph.get_tensor(tensor_name);
        if (!tensor)
            tensor = graph.create_tensor(tensor_name);
        graph.output_tensors.push_back(tensor);
    }
}

void ONNXParser::parse_nodes(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph) {
    for (int i = 0; i < onnx_graph.node_size(); ++i) {
        const onnx::NodeProto& node_proto = onnx_graph.node(i);

        std::string node_name = node_proto.name();
        if (node_name.empty())
            node_name = node_proto.op_type() + "_" + std::to_string(i);
        
        std::string op_type_str = node_proto.op_type();
        OpType op_type = convert_op_type(op_type_str);
        
        if (op_type == OpType::UNKNOWN) {
            dbgs << "Warning: Skipping unsupported operation: " << op_type_str << std::endl;
            continue;
        }

        std::string domain = node_proto.domain();

        Node* node = graph.create_node(node_name, op_type);
        node->domain = domain;

        for (int j = 0; j < node_proto.input_size(); ++j) {
            std::string input_name = node_proto.input(j);
            if (!input_name.empty()) {
                Tensor* tensor = graph.get_tensor(input_name);
                if (!tensor)
                    tensor = graph.create_tensor(input_name);
                node->inputs.push_back(tensor);
            }
        }

        for (int j = 0; j < node_proto.output_size(); ++j) {
            std::string output_name = node_proto.output(j);
            if (!output_name.empty()) {
                Tensor* tensor = graph.get_tensor(output_name);
                if (!tensor)
                    tensor = graph.create_tensor(output_name);
                node->outputs.push_back(tensor);
            }
        }

        for (int j = 0; j < node_proto.attribute_size(); ++j) {
            const onnx::AttributeProto& attr_proto = node_proto.attribute(j);
            Attribute attr = graph.create_attribute(attr_proto.name(), attr_proto);
            node->attributes[attr.name] = std::move(attr);
        }
    }
}

ComputationalGraph ONNXParser::parse(const std::string& model_path) {
    dbgs << "Parsing ONNX model from: " << model_path << std::endl;

    std::ifstream file(model_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open ONNX file: " + model_path);
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(file_size);
    file.read(buffer.data(), file_size);
    file.close();

    return parse_from_buffer(buffer);
}

ComputationalGraph ONNXParser::parse_from_buffer(const std::vector<char>& buffer) {
    ComputationalGraph graph;

    onnx::ModelProto model;
    if (!model.ParseFromArray(buffer.data(), static_cast<int>(buffer.size()))) {
        throw std::runtime_error("Failed to parse ONNX model from buffer");
    }

    const onnx::GraphProto& onnx_graph = model.graph();
    graph.name = onnx_graph.name();

    dbgs << "Parsing initializers..." << std::endl;
    parse_initializers(graph, onnx_graph);

    dbgs << "Parsing inputs..." << std::endl;
    parse_inputs(graph, onnx_graph);

    dbgs << "Parsing outputs..." << std::endl;
    parse_outputs(graph, onnx_graph);

    dbgs << "Parsing nodes..." << std::endl;
    parse_nodes(graph, onnx_graph);

    dbgs << "ONNX parsing completed successfully!" << std::endl;
    return graph;
}
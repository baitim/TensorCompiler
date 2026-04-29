#include "Common/Errors.hpp"
#include "Frontend/Parser.hpp"
#include <format>
#include <fstream>
#include <span>
#include <onnx/onnx_pb.h>

namespace tc::fe {

class ErrorONNXParse : public Error {
public:
    ErrorONNXParse(std::string_view reason)
        : Error(std::format("ONNX parsing error: {}", reason)) {}
};

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

OpType ONNXParser::convert_op_type(std::string_view op_type_str) {
    if (op_type_str == "Add") return OpType::ADD;
    if (op_type_str == "Mul") return OpType::MUL;
    if (op_type_str == "Conv") return OpType::CONV;
    if (op_type_str == "Relu") return OpType::RELU;
    if (op_type_str == "MatMul") return OpType::MATMUL;
    if (op_type_str == "Gemm") return OpType::GEMM;
    if (op_type_str == "Reshape") return OpType::RESHAPE;
    return OpType::UNKNOWN;
}

void ONNXParser::parse_value_info(ComputationalGraph& graph,
                                  const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& value_infos,
                                  bool is_input) {
    for (int i = 0; i < value_infos.size(); ++i) {
        const onnx::ValueInfoProto& value_info = value_infos.Get(i);
        std::string tensor_name = value_info.name();
        auto tensor_opt = graph.get_tensor(tensor_name);
        Tensor* tensor = tensor_opt.value_or(nullptr);
        if (!tensor)
            tensor = graph.create_tensor(tensor_name);

        if (value_info.has_type() && value_info.type().has_tensor_type()) {
            const auto& tensor_type = value_info.type().tensor_type();
            DataType dtype = convert_onnx_type(tensor_type.elem_type());
            tensor->dtype = dtype;

            if (tensor_type.has_shape()) {
                const auto& shape_proto = tensor_type.shape();
                tensor->shape.clear();
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

        if (is_input)
            graph.add_input_tensor(tensor);
        else
            graph.add_output_tensor(tensor);
    }
}

static void fill_tensor_data(Tensor* tensor, const onnx::TensorProto& tensor_proto) {
    switch (tensor->dtype) {
        case DataType::FLOAT: {
            std::vector<float> data;
            if (tensor_proto.has_raw_data()) {
                const std::string& raw = tensor_proto.raw_data();
                const float* ptr = reinterpret_cast<const float*>(raw.data());
                data.assign(ptr, ptr + raw.size() / sizeof(float));
            } else {
                data.reserve(tensor_proto.float_data_size());
                for (int i = 0; i < tensor_proto.float_data_size(); ++i)
                    data.push_back(tensor_proto.float_data(i));
            }
            tensor->data = std::move(data);
            break;
        }
        case DataType::INT32: {
            std::vector<int32_t> data;
            if (tensor_proto.has_raw_data()) {
                const std::string& raw = tensor_proto.raw_data();
                const int32_t* ptr = reinterpret_cast<const int32_t*>(raw.data());
                data.assign(ptr, ptr + raw.size() / sizeof(int32_t));
            } else {
                data.reserve(tensor_proto.int32_data_size());
                for (int i = 0; i < tensor_proto.int32_data_size(); ++i)
                    data.push_back(tensor_proto.int32_data(i));
            }
            tensor->data = std::move(data);
            break;
        }
        case DataType::INT64: {
            std::vector<int64_t> data;
            if (tensor_proto.has_raw_data()) {
                const std::string& raw = tensor_proto.raw_data();
                const int64_t* ptr = reinterpret_cast<const int64_t*>(raw.data());
                data.assign(ptr, ptr + raw.size() / sizeof(int64_t));
            } else {
                data.reserve(tensor_proto.int64_data_size());
                for (int i = 0; i < tensor_proto.int64_data_size(); ++i)
                    data.push_back(tensor_proto.int64_data(i));
            }
            tensor->data = std::move(data);
            break;
        }
        case DataType::UINT8: {
            std::vector<uint8_t> data;
            if (tensor_proto.has_raw_data()) {
                const std::string& raw = tensor_proto.raw_data();
                data.assign(raw.begin(), raw.end());
            } else {
                data.reserve(tensor_proto.int32_data_size());
                for (int i = 0; i < tensor_proto.int32_data_size(); ++i)
                    data.push_back(static_cast<uint8_t>(tensor_proto.int32_data(i)));
            }
            tensor->data = std::move(data);
            break;
        }
        default:
            break;
    }
}

void ONNXParser::parse_initializers(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph) {
    for (const auto& tensor_proto : onnx_graph.initializer()) {
        std::string tensor_name = tensor_proto.name();

        std::vector<int64_t> shape;
        shape.reserve(tensor_proto.dims_size());
        for (int j = 0; j < tensor_proto.dims_size(); ++j)
            shape.push_back(tensor_proto.dims(j));

        DataType dtype = convert_onnx_type(tensor_proto.data_type());

        Tensor* tensor = graph.create_tensor_with_data(tensor_name, shape, dtype);
        fill_tensor_data(tensor, tensor_proto);
        tensor->is_initializer = true;
        graph.add_initializer(tensor);
    }
}

void ONNXParser::parse_inputs(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph) {
    parse_value_info(graph, onnx_graph.input(), true);
}

void ONNXParser::parse_outputs(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph) {
    parse_value_info(graph, onnx_graph.output(), false);
}

void ONNXParser::parse_attributes(Node* node, const onnx::NodeProto& node_proto) {
    for (int j = 0; j < node_proto.attribute_size(); ++j) {
        const onnx::AttributeProto& attr_proto = node_proto.attribute(j);
        std::unique_ptr<AttrValueBase> value;

        switch (attr_proto.type()) {
            case onnx::AttributeProto_AttributeType_FLOAT:
                value = std::make_unique<FloatAttrValue>(attr_proto.f());
                break;
            case onnx::AttributeProto_AttributeType_INT:
                value = std::make_unique<IntAttrValue>(static_cast<int64_t>(attr_proto.i()));
                break;
            case onnx::AttributeProto_AttributeType_STRING:
                value = std::make_unique<StringAttrValue>(attr_proto.s());
                break;
            case onnx::AttributeProto_AttributeType_FLOATS: {
                std::vector<float> floats;
                floats.reserve(attr_proto.floats_size());
                for (int k = 0; k < attr_proto.floats_size(); ++k)
                    floats.push_back(attr_proto.floats(k));
                value = std::make_unique<FloatsAttrValue>(floats);
                break;
            }
            case onnx::AttributeProto_AttributeType_INTS: {
                std::vector<int64_t> ints;
                ints.reserve(attr_proto.ints_size());
                for (int k = 0; k < attr_proto.ints_size(); ++k)
                    ints.push_back(attr_proto.ints(k));
                value = std::make_unique<IntsAttrValue>(ints);
                break;
            }
            case onnx::AttributeProto_AttributeType_STRINGS: {
                std::vector<std::string> strings;
                strings.reserve(attr_proto.strings_size());
                for (int k = 0; k < attr_proto.strings_size(); ++k)
                    strings.push_back(attr_proto.strings(k));
                value = std::make_unique<StringsAttrValue>(strings);
                break;
            }
            default:
                value = std::make_unique<IntAttrValue>(static_cast<int64_t>(0));
                break;
        }

        node->add_attribute(attr_proto.name(), std::move(value));
    }
}

void ONNXParser::parse_nodes(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph) {
    const auto& nodes = onnx_graph.node();
    for (int i = 0; i < nodes.size(); ++i) {
        const onnx::NodeProto& node_proto = nodes.Get(i);

        std::string node_name = node_proto.name();
        if (node_name.empty())
            node_name = std::format("{}_{}", node_proto.op_type(), i);

        std::string op_type_str = node_proto.op_type();
        OpType op_type = convert_op_type(op_type_str);

        if (op_type == OpType::UNKNOWN) {
            tc_dbgs << std::format("Warning: Skipping unsupported operation: {}\n", op_type_str);
            continue;
        }

        Node* node = graph.create_node(node_name, op_type);
        node->domain = node_proto.domain();

        for (int j = 0; j < node_proto.input_size(); ++j) {
            std::string input_name = node_proto.input(j);
            if (!input_name.empty()) {
                auto tensor_opt = graph.get_tensor(input_name);
                Tensor* tensor = tensor_opt.value_or(nullptr);
                if (!tensor)
                    tensor = graph.create_tensor(input_name);
                node->inputs.push_back(tensor);
            }
        }

        for (int j = 0; j < node_proto.output_size(); ++j) {
            std::string output_name = node_proto.output(j);
            if (!output_name.empty()) {
                auto tensor_opt = graph.get_tensor(output_name);
                Tensor* tensor = tensor_opt.value_or(nullptr);
                if (!tensor)
                    tensor = graph.create_tensor(output_name);
                node->outputs.push_back(tensor);
            }
        }

        parse_attributes(node, node_proto);
    }
}

ComputationalGraph ONNXParser::parse(std::string_view model_path) {
    tc_dbgs << std::format("Parsing ONNX model from: {}\n", model_path);

    std::ifstream file(std::string(model_path), std::ios::binary);
    if (!file.is_open())
        throw ErrorONNXParse(std::format("Failed to open ONNX file: {}", model_path));

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(file_size);
    file.read(buffer.data(), file_size);
    file.close();

    return parse_from_buffer(std::span(buffer.data(), buffer.size()));
}

ComputationalGraph ONNXParser::parse_from_buffer(std::span<const char> buffer) {
    onnx::ModelProto model;
    if (!model.ParseFromArray(buffer.data(), static_cast<int>(buffer.size())))
        throw ErrorONNXParse("Failed to parse ONNX model from buffer");

    const onnx::GraphProto& onnx_graph = model.graph();
    ComputationalGraph graph(onnx_graph.name());

    tc_dbgs << std::format("Parsing initializers...\n");
    parse_initializers(graph, onnx_graph);

    tc_dbgs << std::format("Parsing inputs...\n");
    parse_inputs(graph, onnx_graph);

    tc_dbgs << std::format("Parsing outputs...\n");
    parse_outputs(graph, onnx_graph);

    tc_dbgs << std::format("Parsing nodes...\n");
    parse_nodes(graph, onnx_graph);

    tc_dbgs << std::format("ONNX parsing completed successfully!\n");
    return graph;
}

} // namespace tc::fe
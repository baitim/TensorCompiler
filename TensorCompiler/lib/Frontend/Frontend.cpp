#include "Frontend.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <onnx/onnx_pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

std::string op_type_to_string(OpType op_type) {
    switch(op_type) {
        case OpType::ADD: return "Add";
        case OpType::MUL: return "Mul";
        case OpType::CONV: return "Conv";
        case OpType::RELU: return "Relu";
        case OpType::MATMUL: return "MatMul";
        case OpType::GEMM: return "Gemm";
        case OpType::RESHAPE: return "Reshape";
        default: return "Unknown";
    }
}

std::string dtype_to_string(DataType dtype) {
    switch(dtype) {
        case DataType::FLOAT: return "FLOAT";
        case DataType::INT32: return "INT32";
        case DataType::INT64: return "INT64";
        case DataType::BOOL: return "BOOL";
        case DataType::STRING: return "STRING";
        case DataType::UINT8: return "UINT8";
        case DataType::DOUBLE: return "DOUBLE";
        default: return "UNKNOWN";
    }
}

void FloatAttrValue::print(std::ostream& os) const { os << value; }
std::string FloatAttrValue::type_name() const { return "FLOAT"; }

void IntAttrValue::print(std::ostream& os) const { os << value; }
std::string IntAttrValue::type_name() const { return "INT"; }

void StringAttrValue::print(std::ostream& os) const { os << value; }
std::string StringAttrValue::type_name() const { return "STRING"; }

FloatsAttrValue::FloatsAttrValue(const onnx::AttributeProto& attr) {
    value.reserve(attr.floats_size());
    for (int i = 0; i < attr.floats_size(); ++i) {
        value.push_back(attr.floats(i));
    }
}

void FloatsAttrValue::print(std::ostream& os) const {
    os << "[";
    for (size_t i = 0; i < value.size(); ++i) {
        os << value[i];
        if (i < value.size() - 1) os << ", ";
    }
    os << "]";
}
std::string FloatsAttrValue::type_name() const { return "FLOATS"; }

IntsAttrValue::IntsAttrValue(const onnx::AttributeProto& attr) {
    value.reserve(attr.ints_size());
    for (int i = 0; i < attr.ints_size(); ++i) {
        value.push_back(attr.ints(i));
    }
}

void IntsAttrValue::print(std::ostream& os) const {
    os << "[";
    for (size_t i = 0; i < value.size(); ++i) {
        os << value[i];
        if (i < value.size() - 1) os << ", ";
    }
    os << "]";
}
std::string IntsAttrValue::type_name() const { return "INTS"; }

StringsAttrValue::StringsAttrValue(const onnx::AttributeProto& attr) {
    value.reserve(attr.strings_size());
    for (int i = 0; i < attr.strings_size(); ++i) {
        value.push_back(attr.strings(i));
    }
}

void StringsAttrValue::print(std::ostream& os) const {
    os << "[";
    for (size_t i = 0; i < value.size(); ++i) {
        os << value[i];
        if (i < value.size() - 1) os << ", ";
    }
    os << "]";
}
std::string StringsAttrValue::type_name() const { return "STRINGS"; }

Attribute AttributeStorage::create_attribute(const std::string& name, const onnx::AttributeProto& attr_proto) {
    Attribute attr;
    attr.name = name;

    switch (attr_proto.type()) {
        case onnx::AttributeProto_AttributeType_FLOAT:
            attr.value.reset(create<FloatAttrValue>(attr_proto));
            break;
        case onnx::AttributeProto_AttributeType_INT:
            attr.value.reset(create<IntAttrValue>(attr_proto));
            break;
        case onnx::AttributeProto_AttributeType_STRING:
            attr.value.reset(create<StringAttrValue>(attr_proto));
            break;
        case onnx::AttributeProto_AttributeType_FLOATS:
            attr.value.reset(create<FloatsAttrValue>(attr_proto));
            break;
        case onnx::AttributeProto_AttributeType_INTS:
            attr.value.reset(create<IntsAttrValue>(attr_proto));
            break;
        case onnx::AttributeProto_AttributeType_STRINGS:
            attr.value.reset(create<StringsAttrValue>(attr_proto));
            break;
        default:
            attr.value.reset(create<IntAttrValue>(static_cast<int64_t>(0)));
            break;
    }
    
    return attr;
}

Attribute ComputationalGraph::create_attribute(const std::string& name, const onnx::AttributeProto& attr_proto) {
    return attr_storage_.create_attribute(name, attr_proto);
}

Tensor* ComputationalGraph::create_tensor(const std::string& name) {
    Tensor* tensor = storage_.create_tensor();
    tensor->name = name;
    tensors[name] = tensor;
    return tensor;
}

Tensor* ComputationalGraph::create_tensor_with_data(const std::string& name, 
                                                   const std::vector<int64_t>& shape,
                                                   DataType dtype) {
    Tensor* tensor = storage_.create_tensor();
    tensor->name = name;
    tensor->shape = shape;
    tensor->dtype = dtype;
    tensors[name] = tensor;
    return tensor;
}

Tensor* ComputationalGraph::get_tensor(const std::string& name) const {
    auto it = tensors.find(name);
    return it != tensors.end() ? it->second : nullptr;
}

Node* ComputationalGraph::create_node(const std::string& name, OpType op_type) {
    Node* node = storage_.create_node();
    node->name = name;
    node->op_type = op_type;
    nodes.push_back(node);
    return node;
}

size_t ComputationalGraph::tensor_count() const { 
    return storage_.tensor_count();
}

size_t ComputationalGraph::node_count() const { 
    return storage_.node_count();
}

void ComputationalGraph::print_summary(std::ostream& os) const {
    os << "=== Computational Graph Summary ===" << std::endl;
    os << "Name: " << name << std::endl;
    os << "Nodes: " << node_count() << std::endl;
    os << "Tensors: " << tensor_count() << std::endl;
    os << "Input tensors: " << input_tensors.size() << std::endl;
    os << "Output tensors: " << output_tensors.size() << std::endl;
    os << "Initializers: " << initializers.size() << std::endl;
}

void ComputationalGraph::print_detailed(std::ostream& os) const {
    print_summary(os);
    
    os << "\n=== Detailed Information ===" << std::endl;
    
    os << "\nInput Tensors:" << std::endl;
    for (auto tensor : input_tensors) {
        os << "  - " << tensor->name << " [";
        for (size_t i = 0; i < tensor->shape.size(); ++i) {
            os << tensor->shape[i];
            if (i < tensor->shape.size() - 1) os << ", ";
        }
        os << "]" << std::endl;
    }
    
    os << "\nOutput Tensors:" << std::endl;
    for (auto tensor : output_tensors) {
        os << "  - " << tensor->name << " [";
        for (size_t i = 0; i < tensor->shape.size(); ++i) {
            os << tensor->shape[i];
            if (i < tensor->shape.size() - 1) os << ", ";
        }
        os << "]" << std::endl;
    }
    
    os << "\nInitializers (weights):" << std::endl;
    for (auto tensor : initializers) {
        os << "  - " << tensor->name << " [";
        for (size_t i = 0; i < tensor->shape.size(); ++i) {
            os << tensor->shape[i];
            if (i < tensor->shape.size() - 1) os << ", ";
        }
        os << "] dtype=" << dtype_to_string(tensor->dtype) << std::endl;
    }
    
    os << "\nNodes:" << std::endl;
    for (auto node : nodes) {
        os << "  - " << node->name << " [" << op_type_to_string(node->op_type) << "]" << std::endl;
        os << "    Inputs: ";
        for (auto input : node->inputs)
            os << input->name << " ";

        os << "\n    Outputs: ";
        for (auto output : node->outputs)
            os << output->name << " ";

        os << "\n    Attributes (" << node->attributes.size() << "):" << std::endl;
        for (const auto& [name, attr] : node->attributes) {
            os << "      " << name << " (" << attr.value->type_name() << "): ";
            attr.value->print(os);
            os << std::endl;
        }
    }
}

void traverse_graph(ComputationalGraph& graph, GraphVisitor& visitor) {
    for (auto& entry : graph.tensors)
        visitor.visit_tensor(entry.second);
    for (Node* node : graph.nodes)
        visitor.visit_node(node);
}

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
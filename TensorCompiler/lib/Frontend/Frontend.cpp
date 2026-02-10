#include "Frontend.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <onnx/onnx_pb.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

Tensor* ComputationalGraph::create_tensor(const std::string& name) {
    auto tensor = std::make_unique<Tensor>();
    tensor->name = name;
    Tensor* ptr = tensor.get();
    tensor_storage_.push_back(std::move(tensor));
    tensors[name] = ptr;
    return ptr;
}

Tensor* ComputationalGraph::create_tensor_with_data(const std::string& name, 
                                                   const std::vector<int64_t>& shape,
                                                   DataType dtype) {
    auto tensor = std::make_unique<Tensor>();
    tensor->name = name;
    tensor->shape = shape;
    tensor->dtype = dtype;
    Tensor* ptr = tensor.get();
    tensor_storage_.push_back(std::move(tensor));
    tensors[name] = ptr;
    return ptr;
}

Tensor* ComputationalGraph::get_tensor(const std::string& name) const {
    auto it = tensors.find(name);
    return it != tensors.end() ? it->second : nullptr;
}

Node* ComputationalGraph::create_node(const std::string& name, const std::string& op_type) {
    auto node = std::make_unique<Node>();
    node->name = name;
    node->op_type = op_type;
    Node* ptr = node.get();
    node_storage_.push_back(std::move(node));
    nodes.push_back(ptr);
    return ptr;
}

size_t ComputationalGraph::tensor_count() const { 
    return tensor_storage_.size(); 
}

size_t ComputationalGraph::node_count() const { 
    return node_storage_.size(); 
}

void ComputationalGraph::print_summary() const {
    std::cout << "=== Computational Graph Summary ===" << std::endl;
    std::cout << "Name: " << name << std::endl;
    std::cout << "Nodes: " << node_count() << std::endl;
    std::cout << "Tensors: " << tensor_count() << std::endl;
    std::cout << "Input tensors: " << input_tensors.size() << std::endl;
    std::cout << "Output tensors: " << output_tensors.size() << std::endl;
    std::cout << "Initializers: " << initializers.size() << std::endl;
    
    std::cout << "\nInput Tensors:" << std::endl;
    for (auto tensor : input_tensors) {
        std::cout << "  - " << tensor->name << " [";
        for (size_t i = 0; i < tensor->shape.size(); ++i) {
            std::cout << tensor->shape[i];
            if (i < tensor->shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "\nOutput Tensors:" << std::endl;
    for (auto tensor : output_tensors) {
        std::cout << "  - " << tensor->name << " [";
        for (size_t i = 0; i < tensor->shape.size(); ++i) {
            std::cout << tensor->shape[i];
            if (i < tensor->shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "\nNodes:" << std::endl;
    for (auto node : nodes) {
        std::cout << "  - " << node->name << " (" << node->op_type << ")" << std::endl;
        std::cout << "    Inputs: ";
        for (auto input : node->inputs) {
            std::cout << input->name << " ";
        }
        std::cout << "\n    Outputs: ";
        for (auto output : node->outputs) {
            std::cout << output->name << " ";
        }
        std::cout << std::endl;
    }
}

GraphExecutor::GraphExecutor(ComputationalGraph&& graph) : graph_(std::move(graph)) {}

void GraphExecutor::execute() {
    traverse_graph(graph_, *this);
}

void GraphExecutor::visit_tensor(Tensor* tensor) {
    if (!tensor->is_initializer) {
        std::cout << "Initializing tensor: " << tensor->name << std::endl;
    }
}

void GraphExecutor::visit_node(Node* node) {
    std::cout << "Executing node: " << node->name << " [" << node->op_type << "]" << std::endl;
    
    if (node->op_type == "Conv") {
        execute_conv(node);
    } else if (node->op_type == "Relu") {
        execute_relu(node);
    } else if (node->op_type == "Add") {
        execute_add(node);
    } else if (node->op_type == "Gemm") {
        execute_gemm(node);
    } else if (node->op_type == "BatchNormalization") {
        execute_batch_norm(node);
    } else if (node->op_type == "MaxPool" || node->op_type == "AveragePool") {
        execute_pool(node);
    } else {
        std::cout << "Warning: Unsupported operation type: " << node->op_type << std::endl;
    }
}

void GraphExecutor::execute_conv(Node* node) {
    std::cout << "  Executing Conv operation" << std::endl;
}

void GraphExecutor::execute_relu(Node* node) {
    std::cout << "  Executing ReLU operation" << std::endl;
}

void GraphExecutor::execute_add(Node* node) {
    std::cout << "  Executing Add operation" << std::endl;
}

void GraphExecutor::execute_gemm(Node* node) {
    std::cout << "  Executing GEMM operation" << std::endl;
}

void GraphExecutor::execute_batch_norm(Node* node) {
    std::cout << "  Executing BatchNorm operation" << std::endl;
}

void GraphExecutor::execute_pool(Node* node) {
    std::cout << "  Executing Pool operation: " << node->op_type << std::endl;
}

void traverse_graph(ComputationalGraph& graph, GraphVisitor& visitor) {
    for (auto& entry : graph.tensors) {
        visitor.visit_tensor(entry.second);
    }
    
    for (Node* node : graph.nodes) {
        visitor.visit_node(node);
    }
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

std::string ONNXParser::dtype_to_string(DataType dtype) {
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

void ONNXParser::process_attribute(const std::string& name, 
                                  const std::string& type_str,
                                  const std::string& value_str,
                                  Node* node) {
    Attribute attr;
    attr.name = name;
    
    if (type_str.find("float") != std::string::npos) {
        attr.type = AttrType::FLOAT;
        attr.value = std::stof(value_str);
    } else if (type_str.find("int") != std::string::npos) {
        attr.type = AttrType::INT;
        attr.value = static_cast<int64_t>(std::stoll(value_str));
    } else if (type_str.find("string") != std::string::npos) {
        attr.type = AttrType::STRING;
        attr.value = value_str;
    }
    
    node->attributes[name] = attr;
}

ComputationalGraph ONNXParser::parse(const std::string& model_path) {
    std::cout << "Parsing ONNX model from: " << model_path << std::endl;
    
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
    
    for (int i = 0; i < onnx_graph.initializer_size(); ++i) {
        const onnx::TensorProto& tensor_proto = onnx_graph.initializer(i);
        
        std::string tensor_name = tensor_proto.name();
        
        std::vector<int64_t> shape;
        for (int j = 0; j < tensor_proto.dims_size(); ++j) {
            shape.push_back(tensor_proto.dims(j));
        }
        
        DataType dtype = convert_onnx_type(tensor_proto.data_type());
        
        Tensor* tensor = graph.create_tensor_with_data(tensor_name, shape, dtype);
        tensor->is_initializer = true;
        graph.initializers.push_back(tensor);
    }
    
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
    
    for (int i = 0; i < onnx_graph.output_size(); ++i) {
        const onnx::ValueInfoProto& value_info = onnx_graph.output(i);
        std::string tensor_name = value_info.name();
        
        Tensor* tensor = graph.get_tensor(tensor_name);
        if (!tensor) {
            tensor = graph.create_tensor(tensor_name);
        }
        
        graph.output_tensors.push_back(tensor);
    }
    
    for (int i = 0; i < onnx_graph.node_size(); ++i) {
        const onnx::NodeProto& node_proto = onnx_graph.node(i);
        
        std::string node_name = node_proto.name();
        if (node_name.empty()) {
            node_name = node_proto.op_type() + "_" + std::to_string(i);
        }
        
        std::string op_type = node_proto.op_type();
        std::string domain = node_proto.domain();
        
        Node* node = graph.create_node(node_name, op_type);
        node->domain = domain;
        
        for (int j = 0; j < node_proto.input_size(); ++j) {
            std::string input_name = node_proto.input(j);
            if (!input_name.empty()) {
                Tensor* tensor = graph.get_tensor(input_name);
                if (!tensor) {
                    tensor = graph.create_tensor(input_name);
                }
                node->inputs.push_back(tensor);
            }
        }
        
        for (int j = 0; j < node_proto.output_size(); ++j) {
            std::string output_name = node_proto.output(j);
            if (!output_name.empty()) {
                Tensor* tensor = graph.get_tensor(output_name);
                if (!tensor) {
                    tensor = graph.create_tensor(output_name);
                }
                node->outputs.push_back(tensor);
            }
        }
        
        for (int j = 0; j < node_proto.attribute_size(); ++j) {
            const onnx::AttributeProto& attr_proto = node_proto.attribute(j);
            std::string attr_name = attr_proto.name();
            
            Attribute attr;
            attr.name = attr_name;
            
            switch (attr_proto.type()) {
                case onnx::AttributeProto_AttributeType_FLOAT:
                    attr.type = AttrType::FLOAT;
                    attr.value = attr_proto.f();
                    break;
                case onnx::AttributeProto_AttributeType_INT:
                    attr.type = AttrType::INT;
                    attr.value = static_cast<int64_t>(attr_proto.i());
                    break;
                case onnx::AttributeProto_AttributeType_STRING:
                    attr.type = AttrType::STRING;
                    attr.value = attr_proto.s();
                    break;
                case onnx::AttributeProto_AttributeType_FLOATS: {
                    attr.type = AttrType::FLOATS;
                    std::vector<float> floats;
                    for (int k = 0; k < attr_proto.floats_size(); ++k) {
                        floats.push_back(attr_proto.floats(k));
                    }
                    attr.value = floats;
                    break;
                }
                case onnx::AttributeProto_AttributeType_INTS: {
                    attr.type = AttrType::INTS;
                    std::vector<int64_t> ints;
                    for (int k = 0; k < attr_proto.ints_size(); ++k) {
                        ints.push_back(attr_proto.ints(k));
                    }
                    attr.value = ints;
                    break;
                }
                case onnx::AttributeProto_AttributeType_STRINGS: {
                    attr.type = AttrType::STRINGS;
                    std::vector<std::string> strings;
                    for (int k = 0; k < attr_proto.strings_size(); ++k) {
                        strings.push_back(attr_proto.strings(k));
                    }
                    attr.value = strings;
                    break;
                }
                default:
                    continue;
            }
            
            node->attributes[attr_name] = attr;
        }
    }
    
    return graph;
}
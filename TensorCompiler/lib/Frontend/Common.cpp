#include "Common.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iomanip>

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

void traverse_graph(ComputationalGraph& graph, GraphVisitor& visitor) {
    for (auto& entry : graph.tensors)
        visitor.visit_tensor(entry.second);
    for (Node* node : graph.nodes)
        visitor.visit_node(node);
}
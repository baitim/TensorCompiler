#include "Common/Node.hpp"

namespace tc {

Attribute* Node::create_attribute(const std::string& name, const onnx::AttributeProto& attr_proto) {
    Attribute attr;
    attr.name = name;

    switch (attr_proto.type()) {
        case onnx::AttributeProto_AttributeType_FLOAT:
            attr.value = std::make_unique<FloatAttrValue>(attr_proto);
            break;
        case onnx::AttributeProto_AttributeType_INT:
            attr.value = std::make_unique<IntAttrValue>(attr_proto);
            break;
        case onnx::AttributeProto_AttributeType_STRING:
            attr.value = std::make_unique<StringAttrValue>(attr_proto);
            break;
        case onnx::AttributeProto_AttributeType_FLOATS:
            attr.value = std::make_unique<FloatsAttrValue>(attr_proto);
            break;
        case onnx::AttributeProto_AttributeType_INTS:
            attr.value = std::make_unique<IntsAttrValue>(attr_proto);
            break;
        case onnx::AttributeProto_AttributeType_STRINGS:
            attr.value = std::make_unique<StringsAttrValue>(attr_proto);
            break;
        default:
            attr.value = std::make_unique<IntAttrValue>(static_cast<int64_t>(0));
            break;
    }

    Attribute* stored = attr_storage.create();
    *stored = std::move(attr);
    attributes[stored->name] = stored;
    return stored;
}

Node* NodeManager::create_node(const std::string& name, OpType op_type) {
    Node* node = node_storage_.create();
    node->name = name;
    node->op_type = op_type;
    nodes.push_back(node);
    return node;
}

size_t NodeManager::node_count() const {
    return node_storage_.count();
}

} // namespace tc
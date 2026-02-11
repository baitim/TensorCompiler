#pragma once

#include "Common/Attribute.hpp"
#include "Common/Storage.hpp"
#include "Common/Tensor.hpp"

namespace tc {

struct Node {
    std::string name;
    OpType op_type;
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;
    Storage<Attribute> attr_storage;
    std::unordered_map<std::string, Attribute*> attributes;
    std::string domain = "";

public:
    Attribute* create_attribute(const std::string& name, const onnx::AttributeProto& attr_proto);
};

class NodeManager {
protected:
    Storage<Node> node_storage_;
    std::vector<Node*> nodes;

public:
    Node* create_node(const std::string& name, OpType op_type);
    size_t node_count() const;
};

} // namespace tc
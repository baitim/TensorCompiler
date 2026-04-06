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
    Node() = default;
    Node(const std::string& name, OpType op_type) : name(name), op_type(op_type) {}
    Attribute* add_attribute(const std::string& name, std::unique_ptr<AttrValueBase> value);
};

class NodeManager {
public:
    std::vector<Node*> nodes;

protected:
    Storage<Node> node_storage_;

public:
    Node* create_node(const std::string& name, OpType op_type);
    size_t node_count() const;
};

std::string op_type_to_string(OpType op_type);

} // namespace tc
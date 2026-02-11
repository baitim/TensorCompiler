#include "Common/Node.hpp"

namespace tc {

Attribute* Node::add_attribute(const std::string& name, std::unique_ptr<AttrValueBase> value) {
    Attribute attr;
    attr.name = name;
    attr.value = std::move(value);

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
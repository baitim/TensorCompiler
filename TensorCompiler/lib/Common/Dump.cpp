#include "Common/Graph.hpp"
#include "Common/Node.hpp"
#include <format>
#include <iomanip>
#include <queue>
#include <unordered_map>

namespace tc {

std::vector<const Node*> ComputationalGraph::topologicalOrder() const {
    std::unordered_map<const Node*, int> inDegree;
    std::unordered_map<const Node*, std::vector<const Node*>> adj;
    std::unordered_map<std::string, const Node*> outputToNode;

    for (const auto* node : nodes)
        for (auto* t : node->outputs)
            outputToNode[t->name] = node;

    for (const auto* node : nodes) {
        inDegree[node] = 0;
        for (auto* t : node->inputs) {
            auto it = outputToNode.find(t->name);
            if (it != outputToNode.end() && it->second != node)
                adj[it->second].push_back(node);
        }
    }

    for (const auto& [node, succs] : adj)
        for (auto* succ : succs)
            inDegree[succ]++;

    std::queue<const Node*> q;
    for (const auto* node : nodes)
        if (inDegree[node] == 0) q.push(node);

    std::vector<const Node*> order;
    while (!q.empty()) {
        const Node* node = q.front(); q.pop();
        order.push_back(node);
        for (auto* succ : adj[node])
            if (--inDegree[succ] == 0) q.push(succ);
    }

    if (order.size() != nodes.size())
        return std::vector<const Node*>(nodes.begin(), nodes.end());
    return order;
}

static std::string dtype_to_string(DataType dtype) {
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

void ComputationalGraph::print_summary(std::ostream& os) const {
    os << "=== Computational Graph Summary ===" << '\n';
    os << std::format("Name: {}\n", name_);
    os << std::format("Nodes: {}\n", node_count());
    os << std::format("Tensors: {}\n", tensor_count());
    os << std::format("Input tensors: {}\n", input_tensors.size());
    os << std::format("Output tensors: {}\n", output_tensors.size());
    os << std::format("Initializers: {}\n", initializers.size());
}

void ComputationalGraph::print_detailed(std::ostream& os) const {
    print_summary(os);

    os << "\n=== Detailed Information ===" << '\n';

    os << "\nInput Tensors:" << '\n';
    for (auto tensor : input_tensors) {
        os << "  - " << tensor->name << " [";
        for (size_t i = 0; i < tensor->shape.size(); ++i) {
            os << tensor->shape[i];
            if (i < tensor->shape.size() - 1) os << ", ";
        }
        os << "]" << '\n';
    }

    os << "\nOutput Tensors:" << '\n';
    for (auto tensor : output_tensors) {
        os << "  - " << tensor->name << " [";
        for (size_t i = 0; i < tensor->shape.size(); ++i) {
            os << tensor->shape[i];
            if (i < tensor->shape.size() - 1) os << ", ";
        }
        os << "]" << '\n';
    }

    os << "\nInitializers (weights):" << '\n';
    for (auto tensor : initializers) {
        os << "  - " << tensor->name << " [";
        for (size_t i = 0; i < tensor->shape.size(); ++i) {
            os << tensor->shape[i];
            if (i < tensor->shape.size() - 1) os << ", ";
        }
        os << std::format("] dtype={}\n", dtype_to_string(tensor->dtype));
    }

    os << "\nNodes:" << '\n';
    for (auto node : nodes) {
        os << std::format("  - {} [{}]\n", node->name, op_type_to_string(node->op_type));
        os << "    Inputs: ";
        for (auto input : node->inputs)
            os << input->name << " ";
        os << '\n';
        os << "    Outputs: ";
        for (auto output : node->outputs)
            os << output->name << " ";
        os << '\n';
        os << std::format("    Attributes ({}):\n", node->attributes.size());
        for (const auto& [name, attr] : node->attributes) {
            os << std::format("      {} ({}): ", name, attr->value->type_name());
            attr->value->print(os);
            os << '\n';
        }
    }
}

static std::string escape_id(const std::string& s) {
    std::ostringstream oss;
    oss << std::quoted(s);
    return oss.str();
}

void ComputationalGraph::dump_graphviz(std::ostream& os) const {
    os << "digraph G {\n";
    os << "  rankdir=TB;\n";
    os << "  node [fontname=\"Arial\"];\n";
    os << "  edge [fontname=\"Arial\"];\n\n";

    for (const auto& [name, tensor] : tensors) {
        std::string shape_str;
        for (size_t i = 0; i < tensor->shape.size(); ++i) {
            if (i > 0) shape_str += "×";
            shape_str += std::to_string(tensor->shape[i]);
        }
        std::string label = shape_str.empty() ? name : std::format("{}\\n{}", name, shape_str);
        os << "  " << escape_id(name) << " [label=" << escape_id(label)
           << ", shape=ellipse, style=filled, fillcolor=";
        if (tensor->is_initializer)
            os << "lightblue";
        else
            os << "lightgray";
        os << "];\n";
    }
    os << "\n";

    for (auto node : nodes) {
        std::string opLabel = std::format("{}\\n{}", node->name, op_type_to_string(node->op_type));
        os << "  " << escape_id(node->name) << " [label=" << escape_id(opLabel)
           << ", shape=box, style=filled, fillcolor=lightgreen];\n";
    }
    os << "\n";

    for (auto node : nodes) {
        for (auto input : node->inputs) {
            os << "  " << escape_id(input->name) << " -> " << escape_id(node->name) << ";\n";
        }
    }
    os << "\n";

    for (auto node : nodes) {
        for (auto output : node->outputs) {
            os << "  " << escape_id(node->name) << " -> " << escape_id(output->name) << ";\n";
        }
    }
    os << "\n";

    os << "}\n";
}

} // namespace tc
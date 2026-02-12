#include "Common/Graph.hpp"
#include "Common/Node.hpp"
#include <iomanip>

namespace tc {

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
    os << "=== Computational Graph Summary ===" << std::endl;
    os << "Name: " << name_ << std::endl;
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
            os << "      " << name << " (" << attr->value->type_name() << "): ";
            attr->value->print(os);
            os << std::endl;
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
        std::string label = name;
        std::string shape_str;
        for (size_t i = 0; i < tensor->shape.size(); ++i) {
            if (i > 0) shape_str += "×";
            shape_str += std::to_string(tensor->shape[i]);
        }
        if (!shape_str.empty()) label += "\\n" + shape_str;
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
        std::string opLabel = node->name + "\\n" + op_type_to_string(node->op_type);
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
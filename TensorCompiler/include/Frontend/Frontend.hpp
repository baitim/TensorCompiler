#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <variant>
#include <cstdint>

enum class DataType {
    FLOAT, INT32, INT64, BOOL, STRING, UINT8, DOUBLE, UNKNOWN
};

struct Tensor {
    std::string name;
    std::vector<int64_t> shape;
    DataType dtype;
    
    std::variant<
        std::vector<float>,
        std::vector<int32_t>,
        std::vector<int64_t>,
        std::vector<uint8_t>,
        std::vector<std::string>,
        std::vector<double>
    > data;
    
    bool is_initializer = false;
};

enum class AttrType {
    FLOAT, INT, STRING, TENSOR, GRAPH, FLOATS, INTS, STRINGS, FLOAT_TENSOR
};

struct Attribute {
    std::string name;
    AttrType type;
    
    std::variant<
        float,
        int64_t,
        std::string,
        Tensor*,
        std::vector<float>,
        std::vector<int64_t>,
        std::vector<std::string>,
        std::vector<double>
    > value;
};

struct Node {
    std::string name;
    std::string op_type;
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;
    std::unordered_map<std::string, Attribute> attributes;
    std::string domain = "";
};

class ComputationalGraph {
private:
    std::vector<std::unique_ptr<Tensor>> tensor_storage_;
    std::vector<std::unique_ptr<Node>> node_storage_;
    
public:
    std::string name;
    std::unordered_map<std::string, Tensor*> tensors;
    std::vector<Node*> nodes;
    std::vector<Tensor*> input_tensors;
    std::vector<Tensor*> output_tensors;
    std::vector<Tensor*> initializers;
    
    Tensor* create_tensor(const std::string& name);
    Tensor* create_tensor_with_data(const std::string& name, 
                                   const std::vector<int64_t>& shape,
                                   DataType dtype);
    Tensor* get_tensor(const std::string& name) const;
    Node* create_node(const std::string& name, const std::string& op_type);
    size_t tensor_count() const;
    size_t node_count() const;
    
    void print_summary() const;
};

class GraphVisitor {
public:
    virtual ~GraphVisitor() = default;
    virtual void visit_tensor(Tensor* tensor) = 0;
    virtual void visit_node(Node* node) = 0;
};

class GraphExecutor : public GraphVisitor {
private:
    ComputationalGraph graph_;
    
    void execute_conv(Node* node);
    void execute_relu(Node* node);
    void execute_add(Node* node);
    void execute_gemm(Node* node);
    void execute_batch_norm(Node* node);
    void execute_pool(Node* node);
    
public:
    GraphExecutor(ComputationalGraph&& graph);
    void execute();
    
    void visit_tensor(Tensor* tensor) override;
    void visit_node(Node* node) override;
};

class ONNXParser {
public:
    ComputationalGraph parse(const std::string& model_path);
    ComputationalGraph parse_from_buffer(const std::vector<char>& buffer);
    
private:
    DataType convert_onnx_type(int32_t onnx_type);
    std::string dtype_to_string(DataType dtype);
    void process_attribute(const std::string& name, 
                          const std::string& type_str,
                          const std::string& value_str,
                          Node* node);
};

void traverse_graph(ComputationalGraph& graph, GraphVisitor& visitor);
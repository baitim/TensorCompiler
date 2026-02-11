#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <variant>
#include <cstdint>
#include <iostream>
#include <onnx/onnx_pb.h>

#ifdef DEBUG_DUMP
#define dbgs std::cout
#else
#define dbgs if (false) std::cout
#endif

namespace tc {

enum class DataType {
    FLOAT, INT32, INT64, BOOL, STRING, UINT8, DOUBLE, UNKNOWN
};

enum class OpType {
    ADD, MUL, CONV, RELU, MATMUL, GEMM, RESHAPE, UNKNOWN
};

class AttrValueBase {
public:
    virtual ~AttrValueBase() = default;
    virtual void print(std::ostream& os) const = 0;
    virtual std::string type_name() const = 0;
};

class FloatAttrValue : public AttrValueBase {
public:
    float value;
    FloatAttrValue(float v) : value(v) {}
    FloatAttrValue(const onnx::AttributeProto& attr) : value(attr.f()) {}
    void print(std::ostream& os) const override;
    std::string type_name() const override;
};

class IntAttrValue : public AttrValueBase {
public:
    int64_t value;
    IntAttrValue(int64_t v) : value(v) {}
    IntAttrValue(const onnx::AttributeProto& attr) : value(static_cast<int64_t>(attr.i())) {}
    void print(std::ostream& os) const override;
    std::string type_name() const override;
};

class StringAttrValue : public AttrValueBase {
public:
    std::string value;
    StringAttrValue(const std::string& v) : value(v) {}
    StringAttrValue(const onnx::AttributeProto& attr) : value(attr.s()) {}
    void print(std::ostream& os) const override;
    std::string type_name() const override;
};

class FloatsAttrValue : public AttrValueBase {
public:
    std::vector<float> value;
    FloatsAttrValue(const std::vector<float>& v) : value(v) {}
    FloatsAttrValue(const onnx::AttributeProto& attr);
    void print(std::ostream& os) const override;
    std::string type_name() const override;
};

class IntsAttrValue : public AttrValueBase {
public:
    std::vector<int64_t> value;
    IntsAttrValue(const std::vector<int64_t>& v) : value(v) {}
    IntsAttrValue(const onnx::AttributeProto& attr);
    void print(std::ostream& os) const override;
    std::string type_name() const override;
};

class StringsAttrValue : public AttrValueBase {
public:
    std::vector<std::string> value;
    StringsAttrValue(const std::vector<std::string>& v) : value(v) {}
    StringsAttrValue(const onnx::AttributeProto& attr);
    void print(std::ostream& os) const override;
    std::string type_name() const override;
};

struct Attribute {
    std::string name;
    std::unique_ptr<AttrValueBase> value;
};

class AttributeStorage {
private:
    std::vector<std::unique_ptr<AttrValueBase>> storage_;

public:
    template <typename T, typename... Args>
    T* create(Args&&... args) {
        auto attr = std::make_unique<T>(std::forward<Args>(args)...);
        T* ptr = attr.get();
        storage_.push_back(std::move(attr));
        return ptr;
    }

    Attribute create_attribute(const std::string& name, const onnx::AttributeProto& attr_proto);
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

struct Node {
    std::string name;
    OpType op_type;
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;
    std::unordered_map<std::string, Attribute> attributes;
    std::string domain = "";
};

class Storage {
private:
    std::vector<std::unique_ptr<Tensor>> tensor_storage_;
    std::vector<std::unique_ptr<Node>> node_storage_;

public:
    template <typename... Args>
    Tensor* create_tensor(Args&&... args) {
        auto tensor = std::make_unique<Tensor>();
        Tensor* ptr = tensor.get();
        tensor_storage_.push_back(std::move(tensor));
        return ptr;
    }

    template <typename... Args>
    Node* create_node(Args&&... args) {
        auto node = std::make_unique<Node>();
        Node* ptr = node.get();
        node_storage_.push_back(std::move(node));
        return ptr;
    }

    size_t tensor_count() const { return tensor_storage_.size(); }
    size_t node_count() const { return node_storage_.size(); }

    const std::vector<std::unique_ptr<Tensor>>& get_tensors() const { return tensor_storage_; }
    const std::vector<std::unique_ptr<Node>>& get_nodes() const { return node_storage_; }
};

class ComputationalGraph {
private:
    Storage storage_;
    AttributeStorage attr_storage_;

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
    Node* create_node(const std::string& name, OpType op_type);
    size_t tensor_count() const;
    size_t node_count() const;

    void print_summary(std::ostream& os) const;
    void print_detailed(std::ostream& os) const;

    Attribute create_attribute(const std::string& name, const onnx::AttributeProto& attr_proto);

    Storage& get_storage() { return storage_; }
    const Storage& get_storage() const { return storage_; }
};

class GraphVisitor {
public:
    virtual ~GraphVisitor() = default;
    virtual void visit_tensor(Tensor* tensor) = 0;
    virtual void visit_node(Node* node) = 0;
};

void traverse_graph(ComputationalGraph& graph, GraphVisitor& visitor);

}
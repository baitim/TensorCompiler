#pragma once

#include "Common/Graph.hpp"
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tc {

class Executor {
public:
    using TensorValues = std::unordered_map<std::string, std::vector<float>>;
    using ShapeMap = std::unordered_map<std::string, std::vector<int64_t>>;

    TensorValues run(const ComputationalGraph& graph, const TensorValues& inputs);
    void execute(const ComputationalGraph& graph, const std::string& input_path, std::ostream& os);

    static std::vector<float> load_binary_floats(const std::string& path);

private:
    void eval_add(const Node* node, TensorValues& vals, ShapeMap& shapes);
    void eval_mul(const Node* node, TensorValues& vals, ShapeMap& shapes);
    void eval_relu(const Node* node, TensorValues& vals, ShapeMap& shapes);
    void eval_matmul(const Node* node, TensorValues& vals, ShapeMap& shapes);
    void eval_gemm(const Node* node, TensorValues& vals, ShapeMap& shapes);
    void eval_conv(const Node* node, TensorValues& vals, ShapeMap& shapes);
    void eval_reshape(const Node* node, TensorValues& vals, ShapeMap& shapes,
                      const ComputationalGraph& graph);
};

} // namespace tc

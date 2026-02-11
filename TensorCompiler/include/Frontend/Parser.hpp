#pragma once
#include "Common.hpp"

namespace tc::fe {

class ONNXParser {
public:
    ComputationalGraph parse(const std::string& model_path);

private:
    DataType convert_onnx_type(int32_t onnx_type);
    OpType convert_op_type(const std::string& op_type_str);

    void parse_initializers(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph);
    void parse_inputs(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph);
    void parse_outputs(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph);
    void parse_nodes(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph);

    ComputationalGraph parse_from_buffer(const std::vector<char>& buffer);
};

}
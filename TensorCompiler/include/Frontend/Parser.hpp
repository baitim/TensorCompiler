#pragma once

#include "Common/Graph.hpp"
#include <onnx/onnx_pb.h>

namespace tc::fe {

class ONNXParser {
private:
    DataType convert_onnx_type(int32_t onnx_type);
    OpType convert_op_type(std::string_view op_type_str);

    void parse_value_info(ComputationalGraph& graph,
                          const google::protobuf::RepeatedPtrField<onnx::ValueInfoProto>& value_infos, bool is_input);
    void parse_initializers(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph);
    void parse_inputs(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph);
    void parse_outputs(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph);
    void parse_attributes(Node* node, const onnx::NodeProto& node_proto);
    void parse_nodes(ComputationalGraph& graph, const onnx::GraphProto& onnx_graph);

    ComputationalGraph parse_from_buffer(const std::vector<char>& buffer);

public:
    ComputationalGraph parse(std::string_view model_path);
};

} // namespace tc::fe
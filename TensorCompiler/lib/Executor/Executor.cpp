#include "Executor/Executor.hpp"
#include "Common/Attribute.hpp"
#include "Common/Errors.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <variant>

namespace tc {

std::vector<float> Executor::load_binary_floats(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw Error(std::format("Cannot open input data file: {}", path));
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<float> data(sz / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

void Executor::execute(const ComputationalGraph& graph,
                       const std::string& input_path,
                       std::ostream& os) {
    if (graph.input_tensors.empty())
        throw Error("Graph has no input tensors");

    TensorValues inputs;
    if (!input_path.empty()) {
        if (graph.input_tensors.size() != 1)
            throw Error("--input-data supports single-input models only");
        inputs[graph.input_tensors[0]->name] = load_binary_floats(input_path);
    } else {
        for (auto* t : graph.input_tensors) {
            int64_t n = 1;
            for (auto d : t->shape) n *= d;
            inputs[t->name] = std::vector<float>(n, 1.0f);
        }
    }

    auto outputs = run(graph, inputs);

    for (auto* t : graph.output_tensors) {
        auto it = outputs.find(t->name);
        if (it == outputs.end()) continue;
        os << t->name << ": [";
        const auto& data = it->second;
        for (size_t i = 0; i < data.size(); ++i) {
            if (i > 0) os << ", ";
            os << data[i];
        }
        os << "]\n";
    }
}

static int64_t numElements(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
}

Executor::TensorValues Executor::run(const ComputationalGraph& graph,
                                     const TensorValues& inputs) {
    TensorValues vals;
    ShapeMap shapes;

    for (auto* t : graph.input_tensors) {
        if (auto it = inputs.find(t->name); it != inputs.end()) {
            vals[t->name] = it->second;
            shapes[t->name] = t->shape;
        }
    }

    for (auto* t : graph.initializers) {
        shapes[t->name] = t->shape;
        std::visit([&](auto&& data) {
            using T = std::decay_t<decltype(data)>;
            if constexpr (std::is_same_v<T, std::vector<float>>) {
                vals[t->name] = data;
            } else if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
                vals[t->name] = std::vector<float>(data.begin(), data.end());
            } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                vals[t->name] = std::vector<float>(data.begin(), data.end());
            } else if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
                vals[t->name] = std::vector<float>(data.begin(), data.end());
            } else if constexpr (std::is_same_v<T, std::vector<double>>) {
                vals[t->name] = std::vector<float>(data.begin(), data.end());
            }
        }, t->data);
    }

    for (const auto* node : graph.topologicalOrder()) {
        switch (node->op_type) {
            case OpType::ADD:     eval_add(node, vals, shapes); break;
            case OpType::MUL:     eval_mul(node, vals, shapes); break;
            case OpType::RELU:    eval_relu(node, vals, shapes); break;
            case OpType::MATMUL:  eval_matmul(node, vals, shapes); break;
            case OpType::GEMM:    eval_gemm(node, vals, shapes); break;
            case OpType::CONV:    eval_conv(node, vals, shapes); break;
            case OpType::RESHAPE: eval_reshape(node, vals, shapes, graph); break;
            default:
                std::cerr << std::format("Executor: unsupported op {}\n",
                                         op_type_to_string(node->op_type));
        }
    }

    TensorValues outputs;
    for (auto* t : graph.output_tensors) {
        if (auto it = vals.find(t->name); it != vals.end())
            outputs[t->name] = it->second;
    }
    return outputs;
}

void Executor::eval_add(const Node* node, TensorValues& vals, ShapeMap& shapes) {
    const auto& a = vals.at(node->inputs[0]->name);
    const auto& b = vals.at(node->inputs[1]->name);
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] + b[i];
    vals[node->outputs[0]->name] = std::move(result);
    shapes[node->outputs[0]->name] = shapes.at(node->inputs[0]->name);
}

void Executor::eval_mul(const Node* node, TensorValues& vals, ShapeMap& shapes) {
    const auto& a = vals.at(node->inputs[0]->name);
    const auto& b = vals.at(node->inputs[1]->name);
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] * b[i];
    vals[node->outputs[0]->name] = std::move(result);
    shapes[node->outputs[0]->name] = shapes.at(node->inputs[0]->name);
}

void Executor::eval_relu(const Node* node, TensorValues& vals, ShapeMap& shapes) {
    const auto& a = vals.at(node->inputs[0]->name);
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); ++i)
        result[i] = std::max(0.0f, a[i]);
    vals[node->outputs[0]->name] = std::move(result);
    shapes[node->outputs[0]->name] = shapes.at(node->inputs[0]->name);
}

void Executor::eval_matmul(const Node* node, TensorValues& vals, ShapeMap& shapes) {
    const auto& a = vals.at(node->inputs[0]->name);
    const auto& b = vals.at(node->inputs[1]->name);
    const auto& shapeA = shapes.at(node->inputs[0]->name);
    const auto& shapeB = shapes.at(node->inputs[1]->name);

    int64_t M = shapeA[0], K = shapeA[1], N = shapeB[1];
    std::vector<float> result(M * N, 0.0f);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n)
            for (int64_t k = 0; k < K; ++k)
                result[m * N + n] += a[m * K + k] * b[k * N + n];

    vals[node->outputs[0]->name] = std::move(result);
    shapes[node->outputs[0]->name] = {M, N};
}

void Executor::eval_gemm(const Node* node, TensorValues& vals, ShapeMap& shapes) {
    const auto& a = vals.at(node->inputs[0]->name);
    const auto& b = vals.at(node->inputs[1]->name);
    const auto& shapeA = shapes.at(node->inputs[0]->name);
    const auto& shapeB = shapes.at(node->inputs[1]->name);

    int64_t transA = 0, transB = 0;
    if (auto it = node->attributes.find("transA"); it != node->attributes.end())
        if (auto* iv = dynamic_cast<IntAttrValue*>(it->second->value.get()))
            transA = iv->value;
    if (auto it = node->attributes.find("transB"); it != node->attributes.end())
        if (auto* iv = dynamic_cast<IntAttrValue*>(it->second->value.get()))
            transB = iv->value;

    int64_t M = transA ? shapeA[1] : shapeA[0];
    int64_t K = transA ? shapeA[0] : shapeA[1];
    int64_t N = transB ? shapeB[0] : shapeB[1];

    std::vector<float> result(M * N, 0.0f);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                float av = transA ? a[k * M + m] : a[m * K + k];
                float bv = transB ? b[n * K + k] : b[k * N + n];
                sum += av * bv;
            }
            result[m * N + n] = sum;
        }

    if (node->inputs.size() >= 3) {
        const auto& c = vals.at(node->inputs[2]->name);
        int64_t cLen = numElements(shapes.at(node->inputs[2]->name));
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n)
                result[m * N + n] += c[n % cLen];
    }

    vals[node->outputs[0]->name] = std::move(result);
    shapes[node->outputs[0]->name] = {M, N};
}

void Executor::eval_conv(const Node* node, TensorValues& vals, ShapeMap& shapes) {
    const auto& input = vals.at(node->inputs[0]->name);
    const auto& weight = vals.at(node->inputs[1]->name);
    const auto& shapeIn = shapes.at(node->inputs[0]->name);
    const auto& shapeW = shapes.at(node->inputs[1]->name);

    int64_t N = shapeIn[0], Cin = shapeIn[1], H = shapeIn[2], W = shapeIn[3];
    int64_t Cout = shapeW[0], kH = shapeW[2], kW = shapeW[3];

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> dilations = {1, 1};
    std::vector<int64_t> pads = {0, 0, 0, 0};

    if (auto it = node->attributes.find("strides"); it != node->attributes.end())
        if (auto* sa = dynamic_cast<IntsAttrValue*>(it->second->value.get()))
            if (sa->value.size() >= 2) strides = {sa->value[0], sa->value[1]};
    if (auto it = node->attributes.find("dilations"); it != node->attributes.end())
        if (auto* da = dynamic_cast<IntsAttrValue*>(it->second->value.get()))
            if (da->value.size() >= 2) dilations = {da->value[0], da->value[1]};
    if (auto it = node->attributes.find("pads"); it != node->attributes.end())
        if (auto* pa = dynamic_cast<IntsAttrValue*>(it->second->value.get()))
            if (pa->value.size() >= 4) pads = {pa->value[0], pa->value[1], pa->value[2], pa->value[3]};

    int64_t Hout = (H + pads[0] + pads[2] - dilations[0] * (kH - 1) - 1) / strides[0] + 1;
    int64_t Wout = (W + pads[1] + pads[3] - dilations[1] * (kW - 1) - 1) / strides[1] + 1;

    std::vector<float> output(N * Cout * Hout * Wout, 0.0f);
    for (int64_t n = 0; n < N; ++n)
        for (int64_t co = 0; co < Cout; ++co)
            for (int64_t oh = 0; oh < Hout; ++oh)
                for (int64_t ow = 0; ow < Wout; ++ow) {
                    float sum = 0.0f;
                    for (int64_t ci = 0; ci < Cin; ++ci)
                        for (int64_t kh = 0; kh < kH; ++kh)
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t ih = oh * strides[0] - pads[0] + kh * dilations[0];
                                int64_t iw = ow * strides[1] - pads[1] + kw * dilations[1];
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                sum += input[((n * Cin + ci) * H + ih) * W + iw]
                                     * weight[((co * Cin + ci) * kH + kh) * kW + kw];
                            }
                    output[((n * Cout + co) * Hout + oh) * Wout + ow] = sum;
                }

    if (node->inputs.size() >= 3) {
        const auto& bias = vals.at(node->inputs[2]->name);
        for (int64_t n = 0; n < N; ++n)
            for (int64_t co = 0; co < Cout; ++co)
                for (int64_t oh = 0; oh < Hout; ++oh)
                    for (int64_t ow = 0; ow < Wout; ++ow)
                        output[((n * Cout + co) * Hout + oh) * Wout + ow] += bias[co];
    }

    vals[node->outputs[0]->name] = std::move(output);
    shapes[node->outputs[0]->name] = {N, Cout, Hout, Wout};
}

void Executor::eval_reshape(const Node* node, TensorValues& vals, ShapeMap& shapes,
                             const ComputationalGraph& graph) {
    const auto& input = vals.at(node->inputs[0]->name);
    int64_t totalElems = numElements(shapes.at(node->inputs[0]->name));

    std::vector<int64_t> newShape;
    auto shapeTensorOpt = graph.get_tensor(node->inputs[1]->name);
    if (shapeTensorOpt && *shapeTensorOpt) {
        Tensor* st = *shapeTensorOpt;
        if (st->is_initializer) {
            std::visit([&newShape](auto&& data) {
                using T = std::decay_t<decltype(data)>;
                if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                    newShape.assign(data.begin(), data.end());
                } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                    for (auto v : data) newShape.push_back(static_cast<int64_t>(v));
                } else if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
                    newShape.assign(data.begin(), data.end());
                }
            }, st->data);
        }
    }

    if (newShape.empty()) {
        if (auto it = vals.find(node->inputs[1]->name); it != vals.end())
            for (auto v : it->second)
                newShape.push_back(static_cast<int64_t>(v));
    }

    int64_t inferIdx = -1, knownProd = 1;
    for (int64_t i = 0; i < (int64_t)newShape.size(); ++i) {
        if (newShape[i] == -1) inferIdx = i;
        else knownProd *= newShape[i];
    }
    if (inferIdx >= 0) newShape[inferIdx] = totalElems / knownProd;

    vals[node->outputs[0]->name] = input;
    shapes[node->outputs[0]->name] = std::move(newShape);
}

} // namespace tc

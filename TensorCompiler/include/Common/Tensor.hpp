#pragma once

#include "Common/Storage.hpp"
#include "Common/Types.hpp"
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tc {

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

public:
    Tensor() = default;
    Tensor(const std::string& name) : name(name) {}
    Tensor(const std::string& name, const std::vector<int64_t>& shape, DataType dtype)
        : name(name), shape(shape), dtype(dtype) {}
};

class TensorManager {
protected:
    Storage<Tensor> tensor_storage_;
    std::unordered_map<std::string, Tensor*> tensors;
    std::vector<Tensor*> input_tensors;
    std::vector<Tensor*> output_tensors;
    std::vector<Tensor*> initializers;

public:
    Tensor* create_tensor(const std::string& name);
    Tensor* create_tensor_with_data(const std::string& name,
                                    const std::vector<int64_t>& shape,
                                    DataType dtype);
    std::optional<Tensor*> get_tensor(const std::string& name) const;
    size_t tensor_count() const;

    void add_input_tensor(Tensor* t) { input_tensors.push_back(t); }
    void add_output_tensor(Tensor* t) { output_tensors.push_back(t); }
    void add_initializer(Tensor* t) { initializers.push_back(t); }
};

} // namespace tc
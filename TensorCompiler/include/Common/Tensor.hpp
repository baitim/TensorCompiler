#pragma once

#include "Common/Storage.hpp"
#include "Common/Types.hpp"
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
    Tensor* get_tensor(const std::string& name) const;
    size_t tensor_count() const;
};

} // namespace tc
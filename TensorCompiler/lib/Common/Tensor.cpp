#include "Common/Tensor.hpp"

namespace tc {

Tensor* TensorManager::create_tensor(const std::string& name) {
    Tensor* tensor = tensor_storage_.create(name);
    tensors[name] = tensor;
    return tensor;
}

Tensor* TensorManager::create_tensor_with_data(const std::string& name,
                                              const std::vector<int64_t>& shape,
                                              DataType dtype) {
    Tensor* tensor = tensor_storage_.create(name, shape, dtype);
    tensors[name] = tensor;
    return tensor;
}

Tensor* TensorManager::get_tensor(const std::string& name) const {
    auto it = tensors.find(name);
    return it != tensors.end() ? it->second : nullptr;
}

size_t TensorManager::tensor_count() const {
    return tensor_storage_.count();
}

} // namespace tc
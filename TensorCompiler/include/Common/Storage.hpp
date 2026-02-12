#pragma once

#include <concepts>
#include <memory>
#include <vector>

namespace tc {

template<typename T>
concept Storable = std::default_initializable<T>;

template<Storable T>
class Storage {
private:
    std::vector<std::unique_ptr<T>> storage_;

public:
    T* create() {
        auto obj = std::make_unique<T>();
        T* ptr = obj.get();
        storage_.push_back(std::move(obj));
        return ptr;
    }

    template<typename... Args>
    T* create(Args&&... args) {
        auto obj = std::make_unique<T>(std::forward<Args>(args)...);
        T* ptr = obj.get();
        storage_.push_back(std::move(obj));
        return ptr;
    }

    size_t count() const { return storage_.size(); }
    const std::vector<std::unique_ptr<T>>& get_all() const { return storage_; }
};

} // namespace tc
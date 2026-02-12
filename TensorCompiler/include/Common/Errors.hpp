#pragma once

#include <exception>
#include <string>

namespace tc {

class Error : public std::exception {
    std::string msg_;
public:
    Error(std::string msg) : msg_(std::move(msg)) {}
    const char* what() const noexcept override { return msg_.c_str(); }
};

} // namespace tc
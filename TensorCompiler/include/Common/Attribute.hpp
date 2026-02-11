#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include <onnx/onnx_pb.h>

namespace tc {

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
    StringAttrValue(std::string_view v) : value(v) {}
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

} // namespace tc
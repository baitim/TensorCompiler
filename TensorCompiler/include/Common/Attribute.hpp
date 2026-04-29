#pragma once

#include <memory>
#include <ostream>
#include <string>
#include <vector>

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
    void print(std::ostream& os) const override;
    std::string type_name() const override;
};

class IntAttrValue : public AttrValueBase {
public:
    int64_t value;
    IntAttrValue(int64_t v) : value(v) {}
    void print(std::ostream& os) const override;
    std::string type_name() const override;
};

class StringAttrValue : public AttrValueBase {
public:
    std::string value;
    StringAttrValue(std::string_view v) : value(v) {}
    void print(std::ostream& os) const override;
    std::string type_name() const override;
};

class FloatsAttrValue : public AttrValueBase {
public:
    std::vector<float> value;
    FloatsAttrValue(std::vector<float> v) : value(std::move(v)) {}
    void print(std::ostream& os) const override;
    std::string type_name() const override;
};

class IntsAttrValue : public AttrValueBase {
public:
    std::vector<int64_t> value;
    IntsAttrValue(std::vector<int64_t> v) : value(std::move(v)) {}
    void print(std::ostream& os) const override;
    std::string type_name() const override;
};

class StringsAttrValue : public AttrValueBase {
public:
    std::vector<std::string> value;
    StringsAttrValue(std::vector<std::string> v) : value(std::move(v)) {}
    void print(std::ostream& os) const override;
    std::string type_name() const override;
};

struct Attribute {
    std::string name;
    std::unique_ptr<AttrValueBase> value;

    Attribute() = default;
    Attribute(std::string name, std::unique_ptr<AttrValueBase> value)
        : name(std::move(name)), value(std::move(value)) {}
};

} // namespace tc
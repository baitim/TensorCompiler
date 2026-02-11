#include "Common/Attribute.hpp"

namespace tc {

void FloatAttrValue::print(std::ostream& os) const { os << value; }
std::string FloatAttrValue::type_name() const { return "FLOAT"; }

void IntAttrValue::print(std::ostream& os) const { os << value; }
std::string IntAttrValue::type_name() const { return "INT"; }

void StringAttrValue::print(std::ostream& os) const { os << value; }
std::string StringAttrValue::type_name() const { return "STRING"; }

FloatsAttrValue::FloatsAttrValue(const onnx::AttributeProto& attr) {
    value.reserve(attr.floats_size());
    for (int i = 0; i < attr.floats_size(); ++i) {
        value.push_back(attr.floats(i));
    }
}

void FloatsAttrValue::print(std::ostream& os) const {
    os << "[";
    for (size_t i = 0; i < value.size(); ++i) {
        os << value[i];
        if (i < value.size() - 1) os << ", ";
    }
    os << "]";
}
std::string FloatsAttrValue::type_name() const { return "FLOATS"; }

IntsAttrValue::IntsAttrValue(const onnx::AttributeProto& attr) {
    value.reserve(attr.ints_size());
    for (int i = 0; i < attr.ints_size(); ++i) {
        value.push_back(attr.ints(i));
    }
}

void IntsAttrValue::print(std::ostream& os) const {
    os << "[";
    for (size_t i = 0; i < value.size(); ++i) {
        os << value[i];
        if (i < value.size() - 1) os << ", ";
    }
    os << "]";
}
std::string IntsAttrValue::type_name() const { return "INTS"; }

StringsAttrValue::StringsAttrValue(const onnx::AttributeProto& attr) {
    value.reserve(attr.strings_size());
    for (int i = 0; i < attr.strings_size(); ++i) {
        value.push_back(attr.strings(i));
    }
}

void StringsAttrValue::print(std::ostream& os) const {
    os << "[";
    for (size_t i = 0; i < value.size(); ++i) {
        os << value[i];
        if (i < value.size() - 1) os << ", ";
    }
    os << "]";
}
std::string StringsAttrValue::type_name() const { return "STRINGS"; }

} // namespace tc
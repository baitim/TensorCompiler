#include "Common/Attribute.hpp"

namespace tc {

void FloatAttrValue::print(std::ostream& os) const { os << value; }
std::string FloatAttrValue::type_name() const { return "FLOAT"; }

void IntAttrValue::print(std::ostream& os) const { os << value; }
std::string IntAttrValue::type_name() const { return "INT"; }

void StringAttrValue::print(std::ostream& os) const { os << value; }
std::string StringAttrValue::type_name() const { return "STRING"; }

void FloatsAttrValue::print(std::ostream& os) const {
    os << "[";
    for (size_t i = 0; i < value.size(); ++i) {
        os << value[i];
        if (i < value.size() - 1) os << ", ";
    }
    os << "]";
}
std::string FloatsAttrValue::type_name() const { return "FLOATS"; }

void IntsAttrValue::print(std::ostream& os) const {
    os << "[";
    for (size_t i = 0; i < value.size(); ++i) {
        os << value[i];
        if (i < value.size() - 1) os << ", ";
    }
    os << "]";
}
std::string IntsAttrValue::type_name() const { return "INTS"; }

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
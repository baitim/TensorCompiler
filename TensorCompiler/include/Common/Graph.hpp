#pragma once

#include "Common/Node.hpp"
#include <iosfwd>

namespace tc {

class ComputationalGraph : public TensorManager, public NodeManager {
private:
    std::string name_;

public:
    ComputationalGraph(std::string_view name) : name_(name) {}

    std::string_view getName() const { return name_; }

    void print_summary(std::ostream& os) const;
    void print_detailed(std::ostream& os) const;
    void dump_graphviz(std::ostream& os) const;
};

} // namespace tc
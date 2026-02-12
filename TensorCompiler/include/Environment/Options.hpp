#pragma once

#include "Common/Errors.hpp"
#include <algorithm>
#include <iostream>
#include <memory>
#include <ranges>
#include <unordered_map>
#include <vector>

namespace tc::env {

class ErrorUndeclOption : public Error {
public:
    ErrorUndeclOption(std::string_view option_name)
        : Error(std::string("missing necessary option: ") + std::string(option_name)) {}
};

class Option {
    std::string name_;
    bool is_necessary_;
    bool is_titled_;
    const size_t max_length_ = 20;
    std::string_view description_;

protected:
    bool is_set_ = false;

public:
    Option(std::string_view name, bool is_necessary, bool is_titled, std::string_view description)
        : name_(name), is_necessary_(is_necessary), is_titled_(is_titled), description_(description) {
        if (name.size() > max_length_)
            throw Error(std::string("option name too long: ") + std::string(name));
    }
    virtual ~Option() = default;

    std::string_view name() const noexcept { return name_; }
    bool is_necessary() const noexcept { return is_necessary_; }
    bool is_titled() const noexcept { return is_titled_; }
    bool is_set() const noexcept { return is_set_; }
    size_t max_length() const noexcept { return max_length_; }
    std::string_view description() const noexcept { return description_; }
    virtual bool parse(std::string_view option) = 0;
};

class OptModelFile final : public Option {
    std::string value_;

public:
    OptModelFile() : Option("<model_file>", true, false, "ONNX model file to parse") {}
    const std::string& value() const noexcept { return value_; }
    bool parse(std::string_view option) override {
        value_ = option;
        is_set_ = true;
        return is_set_;
    }
};

class OptHelp final : public Option {
    bool value_ = false;

public:
    OptHelp() : Option("--help", false, true, "show this help message") {}
    bool value() const noexcept { return value_; }
    bool parse(std::string_view option) override {
        if (option == name()) {
            value_ = true;
            is_set_ = true;
        }
        return is_set_;
    }
};

class OptionGraphvizDump final : public Option {
    bool value_ = false;

public:
    OptionGraphvizDump() : Option("--graphviz-dump", false, true, "generate GraphViz DOT file") {}
    bool value() const noexcept { return value_; }
    bool parse(std::string_view option) override {
        if (option == name()) {
            value_ = true;
            is_set_ = true;
        }
        return is_set_;
    }
};

class OptPrintGraph final : public Option {
    bool value_ = false;

public:
    OptPrintGraph() : Option("--print-graph", false, true, "print detailed graph information") {}
    bool value() const noexcept { return value_; }
    bool parse(std::string_view option) override {
        if (option == name()) {
            value_ = true;
            is_set_ = true;
        }
        return is_set_;
    }
};

class Options {
protected:
    std::unordered_map<std::string, std::unique_ptr<Option>> options_;
    std::pair<int, int> cnt_options_; // first = necessary, second = total

private:
    std::pair<int, int> get_cnt_options() const {
        int necessary = 0, total = 0;
        for (const auto& [_, opt] : options_) {
            ++total;
            if (opt->is_necessary()) ++necessary;
        }
        return {necessary, total};
    }

protected:
    void parse_token(std::string_view token) {
        for (auto& [_, opt] : options_) {
            Option& option = *opt;
            if (!option.is_titled() || option.is_set()) continue;
            if (option.parse(token)) return;
        }
        for (auto& [_, opt] : options_) {
            Option& option = *opt;
            if (option.is_titled() || token.substr(0, 2) == "--") continue;
            if (option.is_set()) continue;
            if (option.parse(token)) return;
        }
    }

public:
    Options() {
        options_.emplace("<model_file>", std::make_unique<OptModelFile>());
        options_.emplace("--help", std::make_unique<OptHelp>());
        options_.emplace("--graphviz-dump", std::make_unique<OptionGraphvizDump>());
        options_.emplace("--print-graph", std::make_unique<OptPrintGraph>());
        cnt_options_ = get_cnt_options();
    }
    virtual ~Options() = default;

    void check_valid() const {
        for (const auto& [_, opt] : options_) {
            if (opt->is_necessary() && !opt->is_set())
                throw ErrorUndeclOption{opt->name()};
        }
    }

    std::vector<std::pair<std::string, Option*>> get_sorted_options() const {
        std::vector<std::pair<std::string, Option*>> sorted;
        for (const auto& [name, opt] : options_)
            sorted.emplace_back(name, opt.get());
        std::sort(sorted.begin(), sorted.end());
        return sorted;
    }
};

class OptData final : public Options {
public:
    void parse(int argc, char* argv[]) {
        if (argc < cnt_options_.first + 1)
            throw Error{"Invalid argument: argc = 2, argv[1] = name of file"};

        for (int i = 1; i < argc; ++i)
            parse_token(argv[i]);

        if (help())
            print_help(std::cout);

        check_valid();
    }

    const std::string& model_file() const {
        auto it = options_.find("<model_file>");
        return static_cast<OptModelFile*>(it->second.get())->value();
    }

    bool graphviz_dump() const noexcept {
        auto it = options_.find("--graphviz-dump");
        return static_cast<OptionGraphvizDump*>(it->second.get())->value();
    }

    bool print_graph() const noexcept {
        auto it = options_.find("--print-graph");
        return static_cast<OptPrintGraph*>(it->second.get())->value();
    }

    bool help() const noexcept {
        auto it = options_.find("--help");
        return static_cast<OptHelp*>(it->second.get())->value();
    }

    std::ostream& print_help(std::ostream& os) const {
        os << "Supported options:\n";
        for (const auto& [name, opt] : get_sorted_options()) {
            os << "  " << name;
            int pad = opt->max_length() - static_cast<int>(name.size());
            for (int i = 0; i < pad; ++i) os << " ";
            os << " " << opt->description() << '\n';
        }
        return os;
    }
};

} // namespace tc::env
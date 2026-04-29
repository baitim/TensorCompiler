#pragma once

#include "Common/Errors.hpp"
#include <algorithm>
#include <format>
#include <iostream>
#include <memory>
#include <ranges>
#include <unordered_map>
#include <vector>

namespace tc::env {

class ErrorUndeclOption : public Error {
public:
    ErrorUndeclOption(std::string_view option_name)
        : Error(std::format("missing necessary option: {}", option_name)) {}
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
            throw Error(std::format("option name too long: {}", name));
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

class OptEmitMlir final : public Option {
    bool value_ = false;
public:
    OptEmitMlir() : Option("--emit-mlir", false, true, "emit MLIR text") {}
    bool value() const noexcept { return value_; }
    bool parse(std::string_view option) override {
        if (option == name()) { value_ = true; is_set_ = true; }
        return is_set_;
    }
};

class OptEmitLlvm final : public Option {
    bool value_ = false;
public:
    OptEmitLlvm() : Option("--emit-llvm", false, true, "emit LLVM IR") {}
    bool value() const noexcept { return value_; }
    bool parse(std::string_view option) override {
        if (option == name()) { value_ = true; is_set_ = true; }
        return is_set_;
    }
};

class OptEmitAsm final : public Option {
    bool value_ = false;
public:
    OptEmitAsm() : Option("--emit-asm", false, true, "emit assembly") {}
    bool value() const noexcept { return value_; }
    bool parse(std::string_view option) override {
        if (option == name()) { value_ = true; is_set_ = true; }
        return is_set_;
    }
};

class OptTargetTriple final : public Option {
    std::string value_;
public:
    OptTargetTriple() : Option("--target", false, true, "target triple (e.g. x86_64-unknown-linux-gnu)") {}
    const std::string& value() const noexcept { return value_; }
    bool parse(std::string_view option) override {
        if (option.substr(0, 9) == "--target=") {
            value_ = option.substr(9);
            is_set_ = true;
        }
        return is_set_;
    }
};

class OptOptLevel final : public Option {
    int value_ = 2;
public:
    OptOptLevel() : Option("--opt-level", false, true, "optimization level (0-3)") {}
    int value() const noexcept { return value_; }
    bool parse(std::string_view option) override {
        if (option.substr(0, 11) == "--opt-level=") {
            value_ = std::stoi(std::string(option.substr(11)));
            is_set_ = true;
        }
        return is_set_;
    }
};

class OptOutputFile final : public Option {
    std::string value_;
public:
    OptOutputFile() : Option("--output", false, true, "output file base name") {}
    const std::string& value() const noexcept { return value_; }
    bool parse(std::string_view option) override {
        if (option.substr(0, 9) == "--output=") {
            value_ = option.substr(9);
            is_set_ = true;
        }
        return is_set_;
    }
};

class OptExecute final : public Option {
    bool value_ = false;
public:
    OptExecute() : Option("--execute", false, true, "run the model using the executor") {}
    bool value() const noexcept { return value_; }
    bool parse(std::string_view option) override {
        if (option == name()) { value_ = true; is_set_ = true; }
        return is_set_;
    }
};

class OptInputData final : public Option {
    std::string value_;
public:
    OptInputData() : Option("--input-data", false, true, "binary file with float32 input data") {}
    const std::string& value() const noexcept { return value_; }
    bool parse(std::string_view option) override {
        if (option.substr(0, 13) == "--input-data=") {
            value_ = option.substr(13);
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
        options_.emplace("--execute", std::make_unique<OptExecute>());
        options_.emplace("--input-data", std::make_unique<OptInputData>());
        options_.emplace("--graphviz-dump", std::make_unique<OptionGraphvizDump>());
        options_.emplace("--print-graph", std::make_unique<OptPrintGraph>());
        options_.emplace("--emit-mlir", std::make_unique<OptEmitMlir>());
        options_.emplace("--emit-llvm", std::make_unique<OptEmitLlvm>());
        options_.emplace("--emit-asm", std::make_unique<OptEmitAsm>());
        options_.emplace("--target", std::make_unique<OptTargetTriple>());
        options_.emplace("--opt-level", std::make_unique<OptOptLevel>());
        options_.emplace("--output", std::make_unique<OptOutputFile>());
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

    bool emit_mlir() const noexcept {
        auto it = options_.find("--emit-mlir");
        return static_cast<OptEmitMlir*>(it->second.get())->value();
    }
    bool emit_llvm() const noexcept {
        auto it = options_.find("--emit-llvm");
        return static_cast<OptEmitLlvm*>(it->second.get())->value();
    }
    bool emit_asm() const noexcept {
        auto it = options_.find("--emit-asm");
        return static_cast<OptEmitAsm*>(it->second.get())->value();
    }
    std::string target_triple() const {
        auto it = options_.find("--target");
        return static_cast<OptTargetTriple*>(it->second.get())->value();
    }
    int opt_level() const {
        auto it = options_.find("--opt-level");
        return static_cast<OptOptLevel*>(it->second.get())->value();
    }
    std::string output_filename() const {
        auto it = options_.find("--output");
        return static_cast<OptOutputFile*>(it->second.get())->value();
    }
    bool execute() const noexcept {
        auto it = options_.find("--execute");
        return static_cast<OptExecute*>(it->second.get())->value();
    }
    std::string input_data() const {
        auto it = options_.find("--input-data");
        return static_cast<OptInputData*>(it->second.get())->value();
    }

    bool help() const noexcept {
        auto it = options_.find("--help");
        return static_cast<OptHelp*>(it->second.get())->value();
    }

    std::ostream& print_help(std::ostream& os) const {
        os << "Supported options:\n";
        for (const auto& [name, opt] : get_sorted_options()) {
            os << std::format("  {:<{}} {}\n", name, opt->max_length(), opt->description());
        }
        return os;
    }
};

} // namespace tc::env
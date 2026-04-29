#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "Common/Graph.hpp"
#include "Environment/Options.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
class MLIRContext;
class OpBuilder;
class Value;
class Type;
class DenseElementsAttr;
class AffineMap;
}

namespace llvm {
class LLVMContext;
class Module;
}

namespace tc {

class MlirEmitter {
public:
    struct Options {
        std::string targetTriple = "";
        int optLevel = 2;
        bool emitMlir = false;
        bool emitLlvm = false;
        bool emitAsm = false;
        std::string outputFilename = "";

        Options() = default;
        explicit Options(const tc::env::OptData& opt);
    };

    MlirEmitter(const ComputationalGraph& graph, const Options& opts);
    ~MlirEmitter();

    void emit();

private:
    void buildMLIR();
    void emitNode(const Node* node);
    void lowerToLLVM();
    void outputMLIR();
    void outputLLVM();
    void outputAssembly();

    mlir::Type getMLIRType(DataType dtype, llvm::ArrayRef<int64_t> shape);
    mlir::Value getValue(const std::string& name);
    void setValue(const std::string& name, mlir::Value value);

    void createConstantTensor(const Tensor* tensor);
    mlir::Value reshapeTensor(mlir::Value input, llvm::ArrayRef<int64_t> newShape);
    mlir::Value broadcastBias(mlir::Value bias, llvm::ArrayRef<int64_t> targetShape);
    mlir::Value createZeroTensor(llvm::ArrayRef<int64_t> shape, mlir::Type elementType);

    const ComputationalGraph& graph_;
    Options opts_;
    std::unique_ptr<mlir::MLIRContext> context_;
    std::unique_ptr<mlir::OpBuilder> builder_;
    mlir::ModuleOp module_;
    std::unique_ptr<llvm::LLVMContext> llvmContext_;
    std::unique_ptr<llvm::Module> llvmModule_;
    std::unordered_map<std::string, mlir::Value> valueMap_;
};

} // namespace tc

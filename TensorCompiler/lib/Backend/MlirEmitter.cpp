#include "Backend/MlirEmitter.hpp"
#include "Common/Node.hpp"
#include "Common/Tensor.hpp"
#include <algorithm>
#include <fstream>
#include <set>
#include <queue>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/TensorToLinalg/TensorToLinalg.h>
#include <mlir/Conversion/LinalgToStandard/LinalgToStandard.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Support/Casting.h>

namespace tc {

MlirEmitter::Options::Options(const tc::env::OptData& opt)
    : targetTriple(opt.target_triple()),
      optLevel(opt.opt_level()),
      emitMlir(opt.emit_mlir()),
      emitLlvm(opt.emit_llvm()),
      emitAsm(opt.emit_asm()),
      outputFilename(opt.output_filename()) {}

MlirEmitter::MlirEmitter(const ComputationalGraph& graph, const Options& opts)
    : graph_(graph), opts_(opts) {
    context_ = std::make_unique<mlir::MLIRContext>();
    context_->loadDialect<mlir::func::FuncDialect>();
    context_->loadDialect<mlir::arith::ArithDialect>();
    context_->loadDialect<mlir::tensor::TensorDialect>();
    context_->loadDialect<mlir::linalg::LinalgDialect>();
    context_->loadDialect<mlir::scf::SCFDialect>();
    builder_ = std::make_unique<mlir::OpBuilder>(context_.get());
    module_ = mlir::ModuleOp::create(mlir::UnknownLoc::get(context_.get()), "tensor_compiler");
}

MlirEmitter::~MlirEmitter() = default;

void MlirEmitter::emit() {
    buildMLIR();
    if (opts_.emitMlir) outputMLIR();
    if (opts_.emitLlvm || opts_.emitAsm) {
        lowerToLLVM();
        if (opts_.emitLlvm) outputLLVM();
        if (opts_.emitAsm) outputAssembly();
    }
}

mlir::Type MlirEmitter::getMLIRType(DataType dtype, llvm::ArrayRef<int64_t> shape) {
    mlir::Type elementType;
    switch (dtype) {
        case DataType::FLOAT: elementType = mlir::Float32Type::get(context_.get()); break;
        case DataType::INT32: elementType = mlir::IntegerType::get(context_.get(), 32); break;
        case DataType::INT64: elementType = mlir::IntegerType::get(context_.get(), 64); break;
        case DataType::BOOL: elementType = mlir::IntegerType::get(context_.get(), 1); break;
        default: elementType = mlir::Float32Type::get(context_.get()); break;
    }
    if (shape.empty()) return elementType;
    return mlir::RankedTensorType::get(shape, elementType);
}

std::vector<const Node*> MlirEmitter::topologicalOrder() const {
    std::unordered_map<const Node*, size_t> inDegree;
    std::unordered_map<const Node*, std::vector<const Node*>> adj;
    std::set<const Node*> allNodes(graph_.nodes.begin(), graph_.nodes.end());

    for (const auto* node : allNodes) {
        inDegree[node] = 0;
        for (auto* inputTensor : node->inputs) {
            for (const auto* other : allNodes) {
                if (std::find(other->outputs.begin(), other->outputs.end(), inputTensor) != other->outputs.end()) {
                    adj[other].push_back(node);
                    inDegree[node]++;
                }
            }
        }
    }
    std::queue<const Node*> q;
    for (const auto& [node, deg] : inDegree) if (deg == 0) q.push(node);
    std::vector<const Node*> order;
    while (!q.empty()) {
        const Node* node = q.front(); q.pop();
        order.push_back(node);
        for (const Node* succ : adj[node]) {
            if (--inDegree[succ] == 0) q.push(succ);
        }
    }
    if (order.size() != allNodes.size()) {
        return std::vector<const Node*>(graph_.nodes.begin(), graph_.nodes.end());
    }
    return order;
}

void MlirEmitter::buildMLIR() {
    auto loc = mlir::UnknownLoc::get(context_.get());
    llvm::SmallVector<mlir::Type> inputTypes, outputTypes;
    for (auto* tensor : graph_.input_tensors) {
        inputTypes.push_back(getMLIRType(tensor->dtype, tensor->shape));
    }
    for (auto* tensor : graph_.output_tensors) {
        outputTypes.push_back(getMLIRType(tensor->dtype, tensor->shape));
    }
    auto funcType = builder_->getFunctionType(inputTypes, outputTypes);
    auto func = mlir::func::FuncOp::create(loc, "main", funcType);
    if (func.getBlocks().empty()) {
        func.addEntryBlock();
    }
    builder_->setInsertionPointToStart(&func.getBlocks().front());

    for (size_t i = 0; i < graph_.input_tensors.size(); ++i) {
        setValue(graph_.input_tensors[i]->name, func.getArgument(i));
    }

    for (const auto* node : topologicalOrder()) {
        emitNode(node);
    }

    llvm::SmallVector<mlir::Value> outputVals;
    for (auto* tensor : graph_.output_tensors) {
        outputVals.push_back(getValue(tensor->name));
    }
    builder_->create<mlir::func::ReturnOp>(loc, outputVals);
    module_.push_back(func);
}

void MlirEmitter::emitNode(const Node* node) {
    auto loc = mlir::UnknownLoc::get(context_.get());
    OpType op = node->op_type;
    std::vector<mlir::Value> inputs;
    for (auto* t : node->inputs) inputs.push_back(getValue(t->name));
    mlir::Value result;

    switch (op) {
        case OpType::ADD: {
            result = builder_->create<mlir::arith::AddFOp>(loc, inputs[0], inputs[1]);
            break;
        }
        case OpType::MUL: {
            result = builder_->create<mlir::arith::MulFOp>(loc, inputs[0], inputs[1]);
            break;
        }
        case OpType::RELU: {
            auto zeroAttr = mlir::FloatAttr::get(mlir::Float32Type::get(context_.get()), 0.0);
            auto zero = builder_->create<mlir::arith::ConstantOp>(loc, zeroAttr);
            auto cmp = builder_->create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, inputs[0], zero);
            result = builder_->create<mlir::arith::SelectOp>(loc, cmp, inputs[0], zero);
            break;
        }
        case OpType::MATMUL: {
            auto lhsTy = llvm::cast<mlir::RankedTensorType>(inputs[0].getType());
            auto rhsTy = llvm::cast<mlir::RankedTensorType>(inputs[1].getType());
            auto resultTy = mlir::RankedTensorType::get({lhsTy.getDimSize(0), rhsTy.getDimSize(1)}, lhsTy.getElementType());
            auto empty = builder_->create<mlir::tensor::EmptyOp>(loc, resultTy.getShape(), resultTy.getElementType());
            result = builder_->create<mlir::linalg::MatmulOp>(loc, mlir::ValueRange{inputs[0], inputs[1]}, mlir::ValueRange{empty})
                .getResult(0);
            break;
        }
        case OpType::RESHAPE: {
            auto outputTensor = node->outputs[0];
            mlir::Type outTy = getMLIRType(outputTensor->dtype, outputTensor->shape);
            result = builder_->create<mlir::tensor::ReshapeOp>(loc, outTy, inputs[0], inputs[1]);
            break;
        }
        default:
            return;
    }
    if (result) setValue(node->outputs[0]->name, result);
}

void MlirEmitter::lowerToLLVM() {
    mlir::PassManager pm(context_.get());
    pm.addPass(mlir::createConvertTensorToLinalgPass());
    pm.addPass(mlir::createConvertLinalgToStandardPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());

    if (mlir::failed(pm.run(module_))) {
        llvm::errs() << "Failed to lower to LLVM\n";
        return;
    }
    mlir::registerLLVMDialectTranslation(*context_);
    llvmContext_ = std::make_unique<llvm::LLVMContext>();
    llvmModule_ = mlir::translateModuleToLLVMIR(module_, *llvmContext_);
    if (!llvmModule_) {
        llvm::errs() << "Failed to translate to LLVM IR\n";
    }
}

void MlirEmitter::outputMLIR() {
    std::string filename = opts_.outputFilename.empty() ? "output.mlir" : opts_.outputFilename;
    std::error_code ec;
    llvm::raw_fd_ostream dest(filename, ec);
    if (ec) return;
    module_.print(dest);
}

void MlirEmitter::outputLLVM() {
    if (!llvmModule_) return;
    std::string filename = opts_.outputFilename.empty() ? "output.ll" : opts_.outputFilename;
    std::error_code ec;
    llvm::raw_fd_ostream dest(filename, ec);
    if (ec) return;
    llvmModule_->print(dest, nullptr);
}

void MlirEmitter::outputAssembly() {
    if (!llvmModule_) return;
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    std::string tripleStr = opts_.targetTriple.empty() ? llvm::sys::getDefaultTargetTriple() : opts_.targetTriple;
    llvm::Triple triple(tripleStr);
    llvmModule_->setTargetTriple(triple);

    std::string error;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, error);
    if (!target) {
        llvm::errs() << "Target lookup failed: " << error << "\n";
        return;
    }
    llvm::TargetOptions opt;
    auto rm = std::optional<llvm::Reloc::Model>();
    std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(triple, "generic", "", opt, rm));
    if (!tm) {
        llvm::errs() << "Failed to create target machine\n";
        return;
    }
    std::string filename = opts_.outputFilename.empty() ? "output.s" : opts_.outputFilename;
    std::error_code ec;
    llvm::raw_fd_ostream dest(filename, ec);
    if (ec) return;
    llvm::legacy::PassManager pass;
    if (tm->addPassesToEmitFile(pass, dest, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
        llvm::errs() << "Target machine cannot emit assembly\n";
        return;
    }
    pass.run(*llvmModule_);
}

mlir::Value MlirEmitter::getValue(const std::string& name) {
    auto it = valueMap_.find(name);
    if (it != valueMap_.end()) return it->second;
    return nullptr;
}

void MlirEmitter::setValue(const std::string& name, mlir::Value value) {
    valueMap_[name] = value;
}

}
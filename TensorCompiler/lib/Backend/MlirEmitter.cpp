#include "Backend/MlirEmitter.hpp"
#include "Common/Node.hpp"
#include "Common/Tensor.hpp"
#include <iostream>
#include <variant>

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Verifier.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Conversion/TensorToLinalg/TensorToLinalg.h>
#include <mlir/Conversion/LinalgToStandard/LinalgToStandard.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
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
    mlir::DialectRegistry registry;
    registry.insert<
        mlir::func::FuncDialect,
        mlir::arith::ArithDialect,
        mlir::tensor::TensorDialect,
        mlir::linalg::LinalgDialect,
        mlir::scf::SCFDialect,
        mlir::memref::MemRefDialect
    >();
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::registerAllToLLVMIRTranslations(registry);
    context_ = std::make_unique<mlir::MLIRContext>(registry);
    context_->loadAllAvailableDialects();
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

void MlirEmitter::buildMLIR() {
    if (graph_.input_tensors.empty() || graph_.output_tensors.empty()) {
        llvm::errs() << "Graph has no inputs or outputs\n";
        return;
    }
    auto loc = mlir::UnknownLoc::get(context_.get());
    llvm::SmallVector<mlir::Type> inputTypes, outputTypes;
    for (auto* tensor : graph_.input_tensors)
        inputTypes.push_back(getMLIRType(tensor->dtype, tensor->shape));
    for (auto* tensor : graph_.output_tensors)
        outputTypes.push_back(getMLIRType(tensor->dtype, tensor->shape));

    auto funcType = builder_->getFunctionType(inputTypes, outputTypes);
    auto func = mlir::func::FuncOp::create(loc, "main", funcType);
    if (func.getBlocks().empty())
        func.addEntryBlock();
    builder_->setInsertionPointToStart(&func.getBlocks().front());

    for (size_t i = 0; i < graph_.input_tensors.size(); ++i)
        setValue(graph_.input_tensors[i]->name, func.getArgument(i));

    for (auto* tensor : graph_.initializers) {
        if (!getValue(tensor->name))
            createConstantTensor(tensor);
    }

    for (const auto* node : graph_.topologicalOrder())
        emitNode(node);

    llvm::SmallVector<mlir::Value> outputVals;
    for (auto* tensor : graph_.output_tensors) {
        mlir::Value v = getValue(tensor->name);
        if (!v) {
            llvm::errs() << "Output tensor " << tensor->name << " not produced\n";
            return;
        }
        outputVals.push_back(v);
    }
    mlir::func::ReturnOp::create(*builder_, loc, outputVals);
    module_.push_back(func);

    if (mlir::failed(mlir::verify(module_))) {
        llvm::errs() << "Module verification failed\n";
        return;
    }
}

void MlirEmitter::createConstantTensor(const Tensor* tensor) {
    auto loc = mlir::UnknownLoc::get(context_.get());
    auto rankedType = getMLIRType(tensor->dtype, tensor->shape);
    if (!mlir::isa<mlir::RankedTensorType>(rankedType)) {
        llvm::errs() << "Constant tensor must be ranked\n";
        return;
    }
    auto shapedType = mlir::cast<mlir::ShapedType>(rankedType);
    mlir::DenseElementsAttr attr;

    switch (tensor->dtype) {
        case DataType::FLOAT: {
            const auto& vec = std::get<std::vector<float>>(tensor->data);
            llvm::SmallVector<float> data(vec.begin(), vec.end());
            attr = mlir::DenseElementsAttr::get(shapedType, llvm::ArrayRef(data));
            break;
        }
        case DataType::INT32: {
            const auto& vec = std::get<std::vector<int32_t>>(tensor->data);
            llvm::SmallVector<int32_t> data(vec.begin(), vec.end());
            attr = mlir::DenseElementsAttr::get(shapedType, llvm::ArrayRef(data));
            break;
        }
        case DataType::INT64: {
            std::vector<int64_t> data;
            std::visit([&data](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
                    data.assign(arg.begin(), arg.end());
                } else if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
                    data.assign(arg.begin(), arg.end());
                } else if constexpr (std::is_same_v<T, std::vector<uint8_t>>) {
                    data.assign(arg.begin(), arg.end());
                } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                    data.assign(arg.begin(), arg.end());
                } else if constexpr (std::is_same_v<T, std::vector<double>>) {
                    data.assign(arg.begin(), arg.end());
                }
            }, tensor->data);
            if (data.empty() && shapedType.getNumElements() > 0) {
                llvm::errs() << "No data for INT64 tensor " << tensor->name << "\n";
                return;
            }
            llvm::SmallVector<int64_t> dataVec(data.begin(), data.end());
            attr = mlir::DenseElementsAttr::get(shapedType, llvm::ArrayRef(dataVec));
            break;
        }
        case DataType::UINT8: {
            const auto& vec = std::get<std::vector<uint8_t>>(tensor->data);
            llvm::SmallVector<uint8_t> data(vec.begin(), vec.end());
            attr = mlir::DenseElementsAttr::get(shapedType, llvm::ArrayRef(data));
            break;
        }
        default:
            llvm::errs() << "Unsupported initializer data type for tensor " << tensor->name << "\n";
            return;
    }

    auto constant = mlir::arith::ConstantOp::create(*builder_, loc, attr);
    setValue(tensor->name, constant.getResult());
}

mlir::Value MlirEmitter::reshapeTensor(mlir::Value input, llvm::ArrayRef<int64_t> newShape) {
    auto loc = mlir::UnknownLoc::get(context_.get());
    auto inputTy = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto resultTy = mlir::RankedTensorType::get(newShape, inputTy.getElementType());

    llvm::SmallVector<int64_t> shapeVec(newShape.begin(), newShape.end());
    auto shapeType = mlir::RankedTensorType::get({(int64_t)shapeVec.size()}, builder_->getIntegerType(64));
    auto shapeAttr = mlir::DenseIntElementsAttr::get(shapeType, shapeVec);
    auto shapeConst = mlir::arith::ConstantOp::create(*builder_, loc, shapeAttr);

    return mlir::tensor::ReshapeOp::create(*builder_, loc, resultTy, input, shapeConst);
}

mlir::Value MlirEmitter::createZeroTensor(llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
    auto loc = mlir::UnknownLoc::get(context_.get());
    auto tensorType = mlir::RankedTensorType::get(shape, elementType);
    mlir::DenseElementsAttr zeroAttr;

    if (elementType.isF32()) {
        std::vector<float> zeros(tensorType.getNumElements(), 0.0f);
        zeroAttr = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef(zeros));
    } else if (elementType.isInteger(32)) {
        std::vector<int32_t> zeros(tensorType.getNumElements(), 0);
        zeroAttr = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef(zeros));
    } else {
        std::vector<float> zeros(tensorType.getNumElements(), 0.0f);
        zeroAttr = mlir::DenseElementsAttr::get(tensorType, llvm::ArrayRef(zeros));
    }

    return mlir::arith::ConstantOp::create(*builder_, loc, zeroAttr);
}

mlir::Value MlirEmitter::broadcastBias(mlir::Value bias, llvm::ArrayRef<int64_t> targetShape) {
    auto loc = mlir::UnknownLoc::get(context_.get());
    auto biasTy = mlir::cast<mlir::RankedTensorType>(bias.getType());
    auto resultTy = mlir::RankedTensorType::get(targetShape, biasTy.getElementType());

    int64_t rank = static_cast<int64_t>(targetShape.size());
    int64_t biasRank = biasTy.getRank();

    auto outputInit = createZeroTensor(targetShape, biasTy.getElementType());

    llvm::SmallVector<mlir::AffineExpr> inputExprs;
    for (int64_t i = rank - biasRank; i < rank; ++i)
        inputExprs.push_back(builder_->getAffineDimExpr(i));

    auto inputMap = mlir::AffineMap::get(rank, 0, inputExprs, context_.get());
    auto outputMap = mlir::AffineMap::getMultiDimIdentityMap(rank, context_.get());

    llvm::SmallVector<mlir::utils::IteratorType> iterTypes(rank, mlir::utils::IteratorType::parallel);

    auto generic = mlir::linalg::GenericOp::create(
        *builder_, loc,
        mlir::TypeRange{resultTy},
        mlir::ValueRange{bias},
        mlir::ValueRange{outputInit},
        llvm::ArrayRef<mlir::AffineMap>{inputMap, outputMap},
        iterTypes,
        [](mlir::OpBuilder& b, mlir::Location loc, mlir::ValueRange args) {
            mlir::linalg::YieldOp::create(b, loc, args[0]);
        }
    );

    return generic.getResult(0);
}

void MlirEmitter::emitNode(const Node* node) {
    auto loc = mlir::UnknownLoc::get(context_.get());
    OpType op = node->op_type;
    std::vector<mlir::Value> inputs;
    for (auto* t : node->inputs) {
        mlir::Value v = getValue(t->name);
        if (!v) {
            llvm::errs() << "Missing value for input tensor: " << t->name
                         << " in node " << node->name << "\n";
            return;
        }
        inputs.push_back(v);
    }
    if (node->outputs.empty()) {
        llvm::errs() << "Node " << node->name << " has no outputs\n";
        return;
    }
    mlir::Value result;

    auto toFloat = [&](mlir::Value val) -> mlir::Value {
        auto type = val.getType();
        if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
            if (tensorType.getElementType().isF32()) return val;
            auto newType = mlir::RankedTensorType::get(tensorType.getShape(), builder_->getF32Type());
            return mlir::arith::SIToFPOp::create(*builder_, loc, newType, val);
        }
        if (type.isF32()) return val;
        return val;
    };

    auto isFloatTensor = [](mlir::Value val) -> bool {
        auto type = val.getType();
        if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type))
            return tensorType.getElementType().isF32();
        return type.isF32();
    };

    switch (op) {
        case OpType::ADD: {
            mlir::Value lhs = inputs[0];
            mlir::Value rhs = inputs[1];
            if (isFloatTensor(lhs) || isFloatTensor(rhs)) {
                lhs = toFloat(lhs);
                rhs = toFloat(rhs);
                result = mlir::arith::AddFOp::create(*builder_, loc, lhs, rhs);
            } else {
                result = mlir::arith::AddIOp::create(*builder_, loc, lhs, rhs);
            }
            break;
        }
        case OpType::MUL: {
            mlir::Value lhs = inputs[0];
            mlir::Value rhs = inputs[1];
            if (isFloatTensor(lhs) || isFloatTensor(rhs)) {
                lhs = toFloat(lhs);
                rhs = toFloat(rhs);
                result = mlir::arith::MulFOp::create(*builder_, loc, lhs, rhs);
            } else {
                result = mlir::arith::MulIOp::create(*builder_, loc, lhs, rhs);
            }
            break;
        }
        case OpType::RELU: {
            auto inputType = mlir::cast<mlir::RankedTensorType>(inputs[0].getType());
            if (!inputType.getElementType().isF32()) {
                llvm::errs() << "ReLU expects float tensor\n";
                return;
            }
            auto zeroTensor = createZeroTensor(inputType.getShape(), inputType.getElementType());
            result = mlir::arith::MaximumFOp::create(*builder_, loc, inputs[0], zeroTensor);
            break;
        }
        case OpType::MATMUL: {
            auto lhs = inputs[0];
            auto rhs = inputs[1];
            auto lhsTy = mlir::cast<mlir::RankedTensorType>(lhs.getType());
            auto rhsTy = mlir::cast<mlir::RankedTensorType>(rhs.getType());
            if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2) {
                llvm::errs() << "MatMul expects 2D tensors\n";
                return;
            }
            auto resultTy = mlir::RankedTensorType::get(
                {lhsTy.getDimSize(0), rhsTy.getDimSize(1)}, lhsTy.getElementType());
            auto zeroTensor = createZeroTensor(resultTy.getShape(), resultTy.getElementType());
            auto matmul = mlir::linalg::MatmulOp::create(
                *builder_, loc, mlir::ValueRange{lhs, rhs}, mlir::ValueRange{zeroTensor});
            result = matmul.getResult(0);
            break;
        }
        case OpType::GEMM: {
            mlir::Value A = inputs[0];
            mlir::Value B = inputs[1];
            auto lhsTy = mlir::cast<mlir::RankedTensorType>(A.getType());
            auto rhsTy = mlir::cast<mlir::RankedTensorType>(B.getType());
            if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2) {
                llvm::errs() << "Gemm expects 2D tensors\n";
                return;
            }
            auto resultTy = mlir::RankedTensorType::get(
                {lhsTy.getDimSize(0), rhsTy.getDimSize(1)}, lhsTy.getElementType());
            auto zeroTensor = createZeroTensor(resultTy.getShape(), resultTy.getElementType());
            auto matmul = mlir::linalg::MatmulOp::create(
                *builder_, loc, mlir::ValueRange{A, B}, mlir::ValueRange{zeroTensor});
            result = matmul.getResult(0);
            if (inputs.size() >= 3) {
                mlir::Value C = inputs[2];
                auto CTy = mlir::cast<mlir::RankedTensorType>(C.getType());
                if (CTy.getShape() != resultTy.getShape())
                    C = broadcastBias(C, resultTy.getShape());
                result = mlir::arith::AddFOp::create(*builder_, loc, result, C);
            }
            break;
        }
        case OpType::CONV: {
            mlir::Value input = inputs[0];
            mlir::Value weight = inputs[1];

            auto it_kernel = node->attributes.find("kernel_shape");
            if (it_kernel == node->attributes.end()) {
                llvm::errs() << "Conv: missing kernel_shape attribute\n";
                return;
            }
            auto* kernelAttr = dynamic_cast<IntsAttrValue*>(it_kernel->second->value.get());
            if (!kernelAttr || kernelAttr->value.size() != 2) {
                llvm::errs() << "Conv: invalid kernel_shape\n";
                return;
            }
            int64_t kh = kernelAttr->value[0];
            int64_t kw = kernelAttr->value[1];

            std::vector<int64_t> strides = {1, 1};
            auto it_strides = node->attributes.find("strides");
            if (it_strides != node->attributes.end()) {
                auto* sa = dynamic_cast<IntsAttrValue*>(it_strides->second->value.get());
                if (sa && sa->value.size() >= 2) strides = {sa->value[0], sa->value[1]};
            }

            std::vector<int64_t> dilations = {1, 1};
            auto it_dil = node->attributes.find("dilations");
            if (it_dil != node->attributes.end()) {
                auto* da = dynamic_cast<IntsAttrValue*>(it_dil->second->value.get());
                if (da && da->value.size() >= 2) dilations = {da->value[0], da->value[1]};
            }

            auto inputTy = mlir::cast<mlir::RankedTensorType>(input.getType());
            auto weightTy = mlir::cast<mlir::RankedTensorType>(weight.getType());
            int64_t N = inputTy.getDimSize(0);
            int64_t H = inputTy.getDimSize(2);
            int64_t W = inputTy.getDimSize(3);
            int64_t F = weightTy.getDimSize(0);
            int64_t Hout = (H - (kh - 1) * dilations[0] - 1) / strides[0] + 1;
            int64_t Wout = (W - (kw - 1) * dilations[1] - 1) / strides[1] + 1;
            llvm::SmallVector<int64_t> outShape = {N, F, Hout, Wout};

            auto zeroTensor = createZeroTensor(outShape, inputTy.getElementType());

            llvm::SmallVector<int64_t, 2> stridesVec(strides.begin(), strides.end());
            llvm::SmallVector<int64_t, 2> dilationsVec(dilations.begin(), dilations.end());
            auto strideAttr = mlir::DenseIntElementsAttr::get(
                mlir::VectorType::get({2}, builder_->getIntegerType(64)),
                llvm::ArrayRef<int64_t>(stridesVec));
            auto dilationAttr = mlir::DenseIntElementsAttr::get(
                mlir::VectorType::get({2}, builder_->getIntegerType(64)),
                llvm::ArrayRef<int64_t>(dilationsVec));

            auto conv = mlir::linalg::Conv2DNchwFchwOp::create(
                *builder_, loc, mlir::TypeRange{zeroTensor.getType()},
                mlir::ValueRange{input, weight}, mlir::ValueRange{zeroTensor},
                strideAttr, dilationAttr);
            result = conv.getResult(0);

            if (inputs.size() >= 3) {
                mlir::Value bias = inputs[2];
                mlir::Value biasBroadcast = broadcastBias(bias, outShape);
                result = mlir::arith::AddFOp::create(*builder_, loc, result, biasBroadcast);
            }
            break;
        }
        case OpType::RESHAPE: {
            if (inputs.size() < 2) {
                llvm::errs() << "Reshape expects shape tensor\n";
                return;
            }
            mlir::Value input = inputs[0];
            mlir::Value shapeTensor = inputs[1];
            auto constOp = shapeTensor.getDefiningOp<mlir::arith::ConstantOp>();
            if (!constOp) {
                llvm::errs() << "Reshape: shape must be a constant\n";
                return;
            }
            auto shapeAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
            if (!shapeAttr) {
                llvm::errs() << "Reshape: shape attribute is not DenseElementsAttr\n";
                return;
            }
            llvm::SmallVector<int64_t> newShape;
            for (auto it = shapeAttr.value_begin<mlir::IntegerAttr>();
                 it != shapeAttr.value_end<mlir::IntegerAttr>(); ++it)
                newShape.push_back((*it).getInt());
            result = reshapeTensor(input, newShape);
            break;
        }
        default:
            llvm::errs() << "Unhandled op type: " << op_type_to_string(op) << "\n";
            return;
    }

    if (result)
        setValue(node->outputs[0]->name, result);
    else
        llvm::errs() << "Failed to create result for node " << node->name << "\n";
}

void MlirEmitter::lowerToLLVM() {
    mlir::PassManager pm(context_.get());

    pm.addPass(mlir::createConvertElementwiseToLinalgPass());

    mlir::bufferization::OneShotBufferizePassOptions bufOpts;
    bufOpts.bufferizeFunctionBoundaries = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());

    if (mlir::failed(pm.run(module_))) {
        llvm::errs() << "Failed to lower to LLVM\n";
        return;
    }
    llvmContext_ = std::make_unique<llvm::LLVMContext>();
    llvmModule_ = mlir::translateModuleToLLVMIR(module_, *llvmContext_);
    if (!llvmModule_)
        llvm::errs() << "Failed to translate to LLVM IR\n";
}

void MlirEmitter::outputMLIR() {
    if (!module_) return;
    if (mlir::failed(mlir::verify(module_))) return;
    std::string base = opts_.outputFilename.empty() ? "output" : opts_.outputFilename;
    std::error_code ec;
    llvm::raw_fd_ostream dest(base + ".mlir", ec);
    if (ec) return;
    module_.print(dest);
    dest.flush();
}

void MlirEmitter::outputLLVM() {
    if (!llvmModule_) return;
    std::string base = opts_.outputFilename.empty() ? "output" : opts_.outputFilename;
    std::error_code ec;
    llvm::raw_fd_ostream dest(base + ".ll", ec);
    if (ec) return;
    llvmModule_->print(dest, nullptr);
}

void MlirEmitter::outputAssembly() {
    if (!llvmModule_) return;
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();

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
    std::unique_ptr<llvm::TargetMachine> tm(
        target->createTargetMachine(triple, "generic", "", opt, rm));
    if (!tm) {
        llvm::errs() << "Failed to create target machine\n";
        return;
    }
    std::string base = opts_.outputFilename.empty() ? "output" : opts_.outputFilename;
    std::error_code ec;
    llvm::raw_fd_ostream dest(base + ".s", ec);
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

} // namespace tc

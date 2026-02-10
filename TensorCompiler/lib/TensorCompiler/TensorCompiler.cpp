#include "Frontend.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <onnx_model_file>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    try {
        ONNXParser parser;
        ComputationalGraph graph = parser.parse(model_path);
        
        graph.print_summary();
        
        GraphExecutor executor(std::move(graph));
        executor.execute();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
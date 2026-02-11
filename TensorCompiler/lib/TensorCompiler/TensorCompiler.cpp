#include "Frontend/Parser.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <onnx_model_file>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    try {
        tc::fe::ONNXParser parser;
        tc::ComputationalGraph graph = parser.parse(model_path);
        
        graph.print_detailed(std::cout);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
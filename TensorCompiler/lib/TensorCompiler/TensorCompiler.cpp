#include "Environment/Options.hpp"
#include "Frontend/Parser.hpp"
#include <fstream>

int main(int argc, char* argv[]) try {
    tc::env::OptData opt;
    opt.parse(argc, argv);

    const std::string& model_path = opt.model_file();

    tc::fe::ONNXParser parser;
    tc::ComputationalGraph graph = parser.parse(model_path);

    if (opt.graphviz_dump()) {
        std::string dot_path = model_path + ".dot";
        std::ofstream dot_file(dot_path);
        if (!dot_file.is_open())
            throw tc::Error("Could not create DOT file: " + dot_path);
        graph.dump_graphviz(dot_file);
        dbgs << "GraphViz DOT file written to: " << dot_path << "\n";
    }

    if (opt.print_graph()) {
        graph.print_detailed(std::cout);
    }

    return 0;

} catch (const tc::Error& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
} catch (...) {
    std::cerr << "Unknown error\n";
    return 1;
}
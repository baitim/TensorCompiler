#include "Environment/Options.hpp"
#include "Frontend/Parser.hpp"
#include <filesystem>
#include <fstream>

int main(int argc, char* argv[]) try {
    tc::env::OptData opt;
    opt.parse(argc, argv);

    const std::string& model_path = opt.model_file();
    std::filesystem::path fs_path(model_path);
    if (!std::filesystem::exists(fs_path))
        throw tc::Error(std::format("Model file does not exist: {}", model_path));

    tc::fe::ONNXParser parser;
    tc::ComputationalGraph graph = parser.parse(model_path);

    if (opt.graphviz_dump()) {
        std::string dot_path = (fs_path.parent_path() / (fs_path.stem().string() + ".dot")).string();
        std::ofstream dot_file(dot_path);
        if (!dot_file.is_open())
            throw tc::Error(std::format("Could not create DOT file: {}", dot_path));
        graph.dump_graphviz(dot_file);
        dbgs << std::format("GraphViz DOT file written to: {}\n", dot_path);
    }

    if (opt.print_graph()) {
        graph.print_detailed(std::cout);
    }

    return 0;

} catch (const tc::Error& error) {
    std::cerr << error.what() << '\n';
    return 1;
} catch (const std::exception& error) {
    std::cerr << std::format("Error: {}\n", error.what());
    return 1;
} catch (...) {
    std::cerr << "Unknown error\n";
    return 1;
}
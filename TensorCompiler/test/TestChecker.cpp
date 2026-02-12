#include "Frontend/Parser.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <gtest/gtest.h>
#include <onnx/onnx_pb.h>

namespace fs = std::filesystem;

class ONNXEndToEndTest : public ::testing::Test {
protected:
    fs::path test_dir;

protected:
    void SetUp() override {
        test_dir = fs::path(__FILE__).parent_path() / "end_to_end";
    }
};

TEST_F(ONNXEndToEndTest, ParseAndCompare) {
    ASSERT_TRUE(fs::exists(test_dir)) << "Test directory not found: " << test_dir;

    int tested = 0;
    for (const auto& entry : fs::directory_iterator(test_dir)) {
        if (entry.path().extension() != ".onnx") continue;
        std::string model_path = entry.path().string();
        std::cout << "Testing: " << model_path << std::endl;

        tc::fe::ONNXParser parser;
        tc::ComputationalGraph graph("test");
        EXPECT_NO_THROW(graph = parser.parse(model_path));

        std::stringstream actual_output;
        graph.print_detailed(actual_output);

        onnx::ModelProto model_proto;
        std::ifstream input(model_path, std::ios::binary);
        ASSERT_TRUE(input.is_open());
        ASSERT_TRUE(model_proto.ParseFromIstream(&input));

        std::string expected_output;
        for (const auto& prop : model_proto.metadata_props()) {
            if (prop.key() == "expected_output") {
                expected_output = prop.value();
                break;
            }
        }
        ASSERT_FALSE(expected_output.empty()) << "No expected_output metadata in " << model_path;

        EXPECT_EQ(actual_output.str(), expected_output);
        ++tested;
    }
    EXPECT_GT(tested, 0) << "No .onnx files found in " << test_dir;
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
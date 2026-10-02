#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

// POSIX System headers for mmap
#include <fcntl.h>

#include "ggml.h"
#include <nlohmann/json.hpp>

#include "TML_ggml.h"
#include "TML_memory.h"

int main(const int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./ToyML <path_to_model.safetensors>\n";
        return EXIT_FAILURE;
    }

    const auto tml_memory = std::make_unique<TML_memory>();
    if (tml_memory->map(argv[1]) != 0) {
        // Handle error or terminate cleanly before accessing buffers
        return EXIT_FAILURE;
    }

    // 4. Parse JSON Header directly from mapped memory (Zero allocation string copies)
    const uint8_t *start = tml_memory->mapped_data + 8;
    const uint8_t *end = start + tml_memory->header_size;
    const nlohmann::json header_json = nlohmann::json::parse(start, end);

    const auto tml_ggml = std::make_unique<TML_ggml>();
    tml_ggml->parse(tml_memory->header_size, header_json);
    tml_ggml->print(tml_memory->mapped_data);

    tml_ggml->build_tree(tml_memory->mapped_data);

    // Query sub-block
    const auto& tree = tml_ggml->get_tree();
    const TML_model_node* layer0_attn = tree.root.find("model.layers.0.self_attn");

    if (layer0_attn) {
        std::cout << "\nFound block: " << layer0_attn->full_path << "\n";
        std::vector<const ggml_tensor*> attn_tensors;
        layer0_attn->collect_tensors(attn_tensors);
        std::cout << "Contains " << attn_tensors.size() << " underlying ggml tensors.\n\n";
    }


    // 1. Inspect the entire model
    const TML_model_node* model_node = tree.root.find("model");
    BlockStats model_stats = tml_ggml->analyze_subblock(model_node);
    std::cout << "Model Total Params: " << model_stats.total_params << "\n";

    // 2. Inspect an entire layer block: model.layers.0
    const TML_model_node* layer0 = tree.root.find("model.layers.0");
    if (layer0) {
        BlockStats l0_stats = tml_ggml->analyze_subblock(layer0);
        std::cout << "Layer 0 Frobenius Norm: " << std::sqrt(l0_stats.Frobenius_norm_sq) << "\n";
    }

    // 3. Inspect a specific sub-module: model.layers.0.input_layernorm
    const TML_model_node* norm_node = tree.root.find("model.layers.0.input_layernorm");
    if (norm_node) {
        BlockStats norm_stats = tml_ggml->analyze_subblock(norm_node);
        std::cout << "LayerNorm Parameter Count: " << norm_stats.total_params << "\n";
    }

    // 4. Inspect the leaf tensor: model.layers.0.input_layernorm.weight
    const TML_model_node* weight_node = tree.root.find("model.layers.0.input_layernorm.weight");
    if (weight_node && weight_node->is_leaf()) {
        std::cout << "Tensor Address: " << weight_node->tensor->data << "\n";
    }


    return EXIT_SUCCESS;
}
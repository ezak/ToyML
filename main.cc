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
    nlohmann::json header_json = nlohmann::json::parse(start, end);

    const auto tml_ggml = std::make_unique<TML_ggml>();
    tml_ggml->parse(tml_memory->header_size, header_json);
    tml_ggml->print(tml_memory->mapped_data);

    return EXIT_SUCCESS;
}
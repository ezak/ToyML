#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <stdexcept>

// POSIX System headers for mmap
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct SafeTensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t data_begin;
    size_t data_end;
};

static enum ggml_type parse_ggml_type(const std::string& dtype) {
    if (dtype == "F32") return GGML_TYPE_F32;
    if (dtype == "F16") return GGML_TYPE_F16;
    if (dtype == "BF16") return GGML_TYPE_BF16;
    if (dtype == "I32") return GGML_TYPE_I32;
    if (dtype == "I16") return GGML_TYPE_I16;
    if (dtype == "I8") return GGML_TYPE_I8;
    throw std::runtime_error("Unsupported dtype: " + dtype);
}

int main(const int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./ToyML <path_to_model.safetensors>\n";
        return 1;
    }

    const std::string file_path = argv[1];

    // 1. Open File Descriptor
    const int fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Failed to open safetensors file: " << file_path << "\n";
        return 1;
    }

    // Get file size
    struct stat st = {};
    if (fstat(fd, &st) == -1) {
        std::cerr << "Failed to stat file.\n";
        close(fd);
        return 1;
    }
    size_t file_size = st.st_size;

    // 2. Memory-map the whole safetensors file
    const auto mapped_data = static_cast<uint8_t*>(
        mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0)
    );

    if (mapped_data == MAP_FAILED) {
        std::cerr << "Failed to mmap file.\n";
        close(fd);
        return 1;
    }

    // Advisory hint for sequential page prefetching during load/inference
    madvise(mapped_data, file_size, MADV_WILLNEED);

    // 3. Read 8-byte header size (uint64_t Little-Endian)
    uint64_t header_size = 0;
    std::memcpy(&header_size, mapped_data, sizeof(uint64_t));

    // 4. Parse JSON Header directly from mapped memory (Zero allocation string copies)
    const auto json_start = reinterpret_cast<const char*>(mapped_data + 8);
    json header_json = json::parse(json_start, json_start + header_size);

    const size_t offset_base = 8 + header_size;
    std::vector<SafeTensorInfo> tensors;

    for (auto& [key, val] : header_json.items()) {
        if (key == "__metadata__") continue;

        SafeTensorInfo info;
        info.name = key;
        info.dtype = val["dtype"].get<std::string>();
        info.shape = val["shape"].get<std::vector<int64_t>>();
        info.data_begin = offset_base + val["data_offsets"][0].get<size_t>();
        info.data_end = offset_base + val["data_offsets"][1].get<size_t>();
        tensors.push_back(info);
    }

    std::cout << "Parsed " << tensors.size() << " tensors from header via mmap.\n";

    // 5. Initialize GGML Context
    const ggml_init_params params = {
        /* .mem_size   = */ .mem_size = ggml_tensor_overhead() * tensors.size() + 1024 * 1024,
        /* .mem_buffer = */ .mem_buffer = nullptr,
        /* .no_alloc   = */ .no_alloc = true,
    };

    struct ggml_context* ctx = ggml_init(params);

    size_t max_name_len = 0;
    for (const auto& t : tensors) {
        max_name_len = std::max(max_name_len, t.name.length());
    }

    // 6. Assign Mapped Pointers directly to GGML Tensors
    for (const auto& t_info : tensors) {
        const enum ggml_type type = parse_ggml_type(t_info.dtype);
        const int ndims = t_info.shape.size();
        ggml_tensor* gtensor = nullptr;

        if (ndims == 1) {
            gtensor = ggml_new_tensor_1d(ctx, type, t_info.shape[0]);
        } else if (ndims == 2) {
            gtensor = ggml_new_tensor_2d(ctx, type, t_info.shape[1], t_info.shape[0]);
        } else if (ndims == 3) {
            gtensor = ggml_new_tensor_3d(ctx, type, t_info.shape[2], t_info.shape[1], t_info.shape[0]);
        } else if (ndims == 4) {
            gtensor = ggml_new_tensor_4d(ctx, type, t_info.shape[3], t_info.shape[2], t_info.shape[1], t_info.shape[0]);
        }

        ggml_set_name(gtensor, t_info.name.c_str());

        gtensor->data = mapped_data + t_info.data_begin;

        std::printf("Mapped Tensor: %-*s | GGML Type: %-6s | Address: %18p | Bytes: %10zu\n",
                    static_cast<int>(max_name_len + 1),
                    gtensor->name,
                    ggml_type_name(gtensor->type),
                    gtensor->data,
                    ggml_nbytes(gtensor));
    }

    // 7. Cleanup
    // Free GGML context metadata (does not touch mmap memory)
    ggml_free(ctx);

    // Unmap memory and close file descriptor
    munmap(mapped_data, file_size);
    close(fd);

    return 0;
}
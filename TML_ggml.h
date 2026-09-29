/*
 * Created by izak on 9/29/26.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef TOYML_TML_GGML_H
#define TOYML_TML_GGML_H

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include <nlohmann/json.hpp>

#include "ggml.h"

struct SafeTensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t data_begin{};
    size_t data_end{};
};

static ggml_type parse_ggml_type(const std::string& dtype) {
    if (dtype == "F32") return GGML_TYPE_F32;
    if (dtype == "F16") return GGML_TYPE_F16;
    if (dtype == "BF16") return GGML_TYPE_BF16;
    if (dtype == "I32") return GGML_TYPE_I32;
    if (dtype == "I16") return GGML_TYPE_I16;
    if (dtype == "I8") return GGML_TYPE_I8;
    throw std::runtime_error("Unsupported dtype: " + dtype);
}

class TML_ggml {

public:
    TML_ggml() = default;
    ~TML_ggml() = default;

    void parse(const uint64_t header_size, const nlohmann::json& header_json) {
        const size_t offset_base = 8 + header_size;

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
    }

    void print(uint8_t* mapped_data) {
        // 5. Initialize GGML Context
        const ggml_init_params params = {
            /* .mem_size   = */ .mem_size = ggml_tensor_overhead() * tensors.size() + 1024 * 1024,
            /* .mem_buffer = */ .mem_buffer = nullptr,
            /* .no_alloc   = */ .no_alloc = true,
        };

        ggml_context* ctx = ggml_init(params);

        size_t max_name_len = 0;
        for (const auto& t : tensors) {
            max_name_len = std::max(max_name_len, t.name.length());
        }

        // 6. Assign Mapped Pointers directly to GGML Tensors
        for (const auto& t_info : tensors) {
            const ggml_type type = parse_ggml_type(t_info.dtype);
            const unsigned long ndims = t_info.shape.size();
            ggml_tensor* gtensor = nullptr;

            if (ndims == 1) {
                gtensor = ggml_new_tensor_1d(ctx, type, t_info.shape[0]);
            } else if (ndims == 2) {
                gtensor = ggml_new_tensor_2d(ctx, type, t_info.shape[1], t_info.shape[0]);
            } else if (ndims == 3) {
                gtensor = ggml_new_tensor_3d(ctx, type, t_info.shape[2], t_info.shape[1], t_info.shape[0]);
            } else if (ndims == 4) {
                gtensor = ggml_new_tensor_4d(ctx, type, t_info.shape[3], t_info.shape[2], t_info.shape[1], t_info.shape[0]);
            } else {
                std::printf("unknown ndims: %lu\n", ndims);
            }

            if (gtensor) {
                ggml_set_name(gtensor, t_info.name.c_str());

                gtensor->data = mapped_data + t_info.data_begin;

                std::printf("Mapped Tensor: %-*s | GGML Type: %-6s | Address: %18p | Bytes: %10zu\n",
                            static_cast<int>(max_name_len + 1),
                            gtensor->name,
                            ggml_type_name(gtensor->type),
                            gtensor->data,
                            ggml_nbytes(gtensor));
            }
        }

        // 7. Cleanup
        // Free GGML context metadata (does not touch mmap memory)
        ggml_free(ctx);
    }

private:
    std::vector<SafeTensorInfo> tensors;
};

#endif //TOYML_TML_GGML_H
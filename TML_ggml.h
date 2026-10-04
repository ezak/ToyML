#ifndef TOYML_TML_GGML_H
#define TOYML_TML_GGML_H

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <future>
#include <numeric>
#include <cmath>
#include <cstring>
#include <nlohmann/json.hpp>

#include "ggml-cpu.h"
#include "ggml.h"

// -----------------------------------------------------------------------------
// SafeTensors & Model Configuration Data Structures
// -----------------------------------------------------------------------------

struct SafeTensorInfo {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    size_t data_begin{};
    size_t data_end{};

    [[nodiscard]] size_t num_elements() const {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), size_t{1},
                               [](const size_t total, const long dim) -> size_t {
                                   return total * static_cast<size_t>(dim);
                               });
    }

    [[nodiscard]] size_t byte_length() const {
        return data_end - data_begin;
    }

    [[nodiscard]] size_t absolute_file_offset(const size_t header_size) const {
        return 8 + header_size + data_begin;
    }
};

struct ModelConfig {
    size_t n_embd = 4096;
    size_t n_head = 32;
    size_t n_head_kv = 32; // For MHA / GQA
    size_t n_layer = 32;
    size_t vocab_size = 32000;
    float norm_eps = 1e-5f;
    size_t max_seq_len = 2048;
};

static ggml_type parse_ggml_type(const std::string &dtype) {
    if (dtype == "F32") return GGML_TYPE_F32;
    if (dtype == "F16") return GGML_TYPE_F16;
    if (dtype == "BF16") return GGML_TYPE_BF16;
    if (dtype == "I32") return GGML_TYPE_I32;
    if (dtype == "I16") return GGML_TYPE_I16;
    if (dtype == "I8") return GGML_TYPE_I8;
    throw std::runtime_error("Unsupported dtype: " + dtype);
}

// Simple Byte-Fallback / Char-level Tokenizer helper for basic demo generation
class SimpleTokenizer {
public:
    static std::vector<int32_t> encode(const std::string &text) {
        std::vector<int32_t> tokens;
        tokens.reserve(text.length());
        for (const char c: text) {
            tokens.push_back(static_cast<uint8_t>(c));
        }
        return tokens;
    }

    static std::string decode(const int32_t token_id) {
        if (token_id >= 0 && token_id < 256) {
            return std::string(1, static_cast<char>(token_id));
        }
        return "[" + std::to_string(token_id) + "]";
    }
};

// -----------------------------------------------------------------------------
// TML_ggml Class Implementation
// -----------------------------------------------------------------------------

class TML_ggml {
public:
    TML_ggml() {
        constexpr ggml_init_params params = {
            .mem_size = 32 * 1024 * 1024,
            .mem_buffer = nullptr,
            .no_alloc = true,
        };

        context_ = ggml_init(params);
        if (!context_) {
            throw std::runtime_error("Failed to initialize persistent GGML context.");
        }
    }

    ~TML_ggml() {
        if (context_) {
            ggml_free(context_);
        }
    }

    void parse(const uint64_t header_size, const nlohmann::json &header_json) {
        const size_t offset_base = 8 + header_size;
        tensors_.reserve(header_json.size());

        for (auto &[key, val]: header_json.items()) {
            if (key == "__metadata__") {
                // Infer config parameters from metadata if present
                if (val.contains("config")) {
                    auto cfg = val["config"];
                    if (cfg.contains("hidden_size")) config_.n_embd = cfg["hidden_size"];
                    if (cfg.contains("num_attention_heads")) config_.n_head = cfg["num_attention_heads"];
                    if (cfg.contains("num_hidden_layers")) config_.n_layer = cfg["num_hidden_layers"];
                    if (cfg.contains("vocab_size")) config_.vocab_size = cfg["vocab_size"];
                }
                continue;
            }

            SafeTensorInfo info;
            info.name = key;
            info.dtype = val["dtype"].get<std::string>();
            info.shape = val["shape"].get<std::vector<int64_t> >();
            info.data_begin = offset_base + val["data_offsets"][0].get<size_t>();
            info.data_end = offset_base + val["data_offsets"][1].get<size_t>();
            tensors_.push_back(info);
        }
    }

    void map_tensors(uint8_t *mapped_data) {
        for (const auto &t_info: tensors_) {
            const ggml_type type = parse_ggml_type(t_info.dtype);
            const size_t ndims = t_info.shape.size();
            ggml_tensor *gtensor = nullptr;

            // GGML dimensions are loaded in reverse order compared to PyTorch/SafeTensors (column-major)
            if (ndims == 1) {
                gtensor = ggml_new_tensor_1d(context_, type, t_info.shape[0]);
            } else if (ndims == 2) {
                gtensor = ggml_new_tensor_2d(context_, type, t_info.shape[1], t_info.shape[0]);
            } else if (ndims == 3) {
                gtensor = ggml_new_tensor_3d(context_, type, t_info.shape[2], t_info.shape[1], t_info.shape[0]);
            } else if (ndims == 4) {
                gtensor = ggml_new_tensor_4d(context_, type, t_info.shape[3], t_info.shape[2], t_info.shape[1],
                                             t_info.shape[0]);
            }

            if (gtensor) {
                ggml_set_name(gtensor, t_info.name.c_str());
                gtensor->data = mapped_data + t_info.data_begin;
                tensor_map_[t_info.name] = gtensor;
            }
        }

        // Auto-detect vocabulary size and embedding dimensions from weights if not explicitly set
        if (auto *embed = get_ggml_tensor("model.embed_tokens.weight")) {
            config_.vocab_size = embed->ne[1];
            config_.n_embd = embed->ne[0];
        }
    }

    [[nodiscard]] ggml_tensor *get_ggml_tensor(const std::string &name) const noexcept {
        const auto it = tensor_map_.find(name);
        return (it != tensor_map_.end()) ? it->second : nullptr;
    }

    [[nodiscard]] ggml_context *get_context() const noexcept { return context_; }
    [[nodiscard]] std::vector<SafeTensorInfo> &get_tensors() noexcept { return tensors_; }
    [[nodiscard]] const ModelConfig &get_config() const noexcept { return config_; }

private:
    std::vector<SafeTensorInfo> tensors_;
    std::unordered_map<std::string, ggml_tensor *> tensor_map_;
    ggml_context *context_ = nullptr;
    mutable ModelConfig config_;
};

#endif // TOYML_TML_GGML_H

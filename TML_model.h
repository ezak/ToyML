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

#ifndef TOYML_TML_MODEL_H
#define TOYML_TML_MODEL_H


#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <functional>
#include "ggml.h"

struct TML_model_node {
    std::string segment_name;      // e.g., "input_layernorm"
    std::string full_path;         // e.g., "model.layers.0.input_layernorm"
    ggml_tensor* tensor = nullptr; // Non-null only at leaf nodes (.weight / .bias)

    // Sub-blocks and tensors under this level
    std::unordered_map<std::string, std::unique_ptr<TML_model_node>> children;

    [[nodiscard]] bool is_leaf() const { return tensor != nullptr; }

    // ------------------------------------------------------------------------
    // Inspection & Traversal API
    // ------------------------------------------------------------------------

    // Navigate to any sub-block: e.g., node->find("model.layers.0")
    const TML_model_node* find(std::string_view path) const {
        if (path.empty()) return this;

        const size_t dot_pos = path.find('.');
        const std::string_view head = (dot_pos == std::string_view::npos) ? path : path.substr(0, dot_pos);
        const std::string_view tail = (dot_pos == std::string_view::npos) ? "" : path.substr(dot_pos + 1);

        const auto it = children.find(std::string(head));
        if (it == children.end()) return nullptr;

        return it->second->find(tail);
    }

    // Recursively aggregate all leaf ggml_tensors contained within this sub-block
    void collect_tensors(std::vector<const ggml_tensor*>& out) const {
        if (tensor) {
            out.push_back(tensor);
        }
        for (const auto& [_, child] : children) {
            child->collect_tensors(out);
        }
    }

    // Apply a visitor callback to every node in this subtree
    void visit(const std::function<void(const TML_model_node&)>& visitor) const {
        visitor(*this);
        for (const auto& [_, child] : children) {
            child->visit(visitor);
        }
    }
};

class TML_model_tree {
public:
    TML_model_node root{.segment_name = "root", .full_path = "root"};

    void insert(const std::string_view full_path, ggml_tensor* gtensor) {
        TML_model_node* current = &root;
        std::string_view remaining = full_path;
        std::string current_accumulated_path;

        while (!remaining.empty()) {
            const size_t dot_pos = remaining.find('.');
            std::string_view segment = (dot_pos == std::string_view::npos) ? remaining : remaining.substr(0, dot_pos);

            if (!current_accumulated_path.empty()) {
                current_accumulated_path += ".";
            }
            current_accumulated_path.append(segment.data(), segment.size());

            std::string seg_key(segment);
            auto& child = current->children[seg_key];
            if (!child) {
                child = std::make_unique<TML_model_node>();
                child->segment_name = seg_key;
                child->full_path = current_accumulated_path;
            }

            current = child.get();

            if (dot_pos == std::string_view::npos) {
                current->tensor = gtensor; // Assign memory pointer at the terminal leaf
                break;
            }

            remaining.remove_prefix(dot_pos + 1);
        }
    }
};

#endif //TOYML_TML_MODEL_H

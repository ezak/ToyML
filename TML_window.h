#ifndef TOYML_TML_WINDOW_H
#define TOYML_TML_WINDOW_H

#include <imgui.h>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cmath>
#include <future>
#include <chrono>

#include "TML_ggml.h"

void show_layers(bool *p_open, const std::vector<SafeTensorInfo> &tensors);

struct TMLMainWindowData {
    ImGuiIO &io;
    TML_ggml &tml_ggml;

    bool show_layers = false;
    bool show_inference_console = false;
};

namespace TML_window_ {
    void show_main_window(ImGuiIO &io, TML_ggml &tml_ggml);
}

inline void TML_window_::show_main_window(ImGuiIO &io, TML_ggml &tml_ggml) {
    IM_ASSERT(ImGui::GetCurrentContext() != nullptr && "Missing Dear ImGui context. Refer to examples app!");
    IMGUI_CHECKVERSION();

    static TMLMainWindowData main_window_data = {.io = io, .tml_ggml = tml_ggml};

    if (main_window_data.show_layers) { show_layers(&main_window_data.show_layers, tml_ggml.get_tensors()); }


    static float f = 0.0f;
    static int counter = 0;
    auto clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    ImGui::Begin("Hello, world!");

    ImGui::Checkbox("Show Layers", &main_window_data.show_layers);
    ImGui::Checkbox("Inference Console", &main_window_data.show_inference_console);

    ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
    ImGui::ColorEdit3("clear color", reinterpret_cast<float *>(&clear_color));

    if (ImGui::Button("Button"))
        counter++;
    ImGui::SameLine();
    ImGui::Text("counter = %d", counter);

    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::End();
}

inline void show_layers(bool *p_open, const std::vector<SafeTensorInfo> &tensors) {
    if (tensors.empty()) {
        ImGui::Text("No tensors available.");
        return;
    }

    static int item_selected_idx = 0;

    if (item_selected_idx >= static_cast<int>(tensors.size())) {
        item_selected_idx = 0;
    }

    ImGui::Begin("Layers", p_open);
    if (ImGui::BeginListBox("##listbox_tensors", ImVec2(-FLT_MIN, 12 * ImGui::GetTextLineHeightWithSpacing()))) {
        for (int n = 0; n < static_cast<int>(tensors.size()); n++) {
            const bool is_selected = (item_selected_idx == n);
            if (ImGui::Selectable(tensors[n].name.c_str(), is_selected)) {
                item_selected_idx = n;
            }
            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndListBox();
    }

    ImGui::Separator();

    const auto &selected = tensors.at(item_selected_idx);
    ImGui::Text("Layer Name: %s", selected.name.c_str());
    ImGui::Spacing();

    if (ImGui::BeginTable("TensorDetails", 2,
                          ImGuiTableFlags_BordersOuter | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("Property", ImGuiTableColumnFlags_WidthFixed, 130.0f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Data Type (dtype)");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "%s", selected.dtype.c_str());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Shape");
        ImGui::TableSetColumnIndex(1);
        std::string shape_str = "[";
        for (size_t i = 0; i < selected.shape.size(); ++i) {
            shape_str += std::to_string(selected.shape[i]);
            if (i + 1 < selected.shape.size()) shape_str += ", ";
        }
        shape_str += "]";
        ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%s", shape_str.c_str());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Total Elements");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%zu", selected.num_elements());

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("Memory Size");
        ImGui::TableSetColumnIndex(1);
        const auto bytes = static_cast<double>(selected.byte_length());
        if (static_cast<double>(selected.byte_length()) >= 1024.0 * 1024.0) {
            ImGui::Text("%.2f MB (%zu bytes)", bytes / (1024.0 * 1024.0), selected.byte_length());
        } else if (bytes >= 1024.0) {
            ImGui::Text("%.2f KB (%zu bytes)", bytes / 1024.0, selected.byte_length());
        } else {
            ImGui::Text("%zu bytes", selected.byte_length());
        }

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("File Offsets");
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("0x%zx -> 0x%zx", selected.data_begin, selected.data_end);

        ImGui::EndTable();
    }

    ImGui::End();
}

#endif // TOYML_TML_WINDOW_H
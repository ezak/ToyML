cmake_minimum_required(VERSION 3.16)
project(ToyML C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# -----------------------------------------------------------------------------
# 1. Find Required System Packages
# -----------------------------------------------------------------------------
find_package(Vulkan REQUIRED)

# -----------------------------------------------------------------------------
# 2. Configure Options BEFORE add_subdirectory
# -----------------------------------------------------------------------------
# GGML Options
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)  # Enable GGML Vulkan backend
set(GGML_VULKAN_SHADERS_GEN OFF CACHE BOOL "" FORCE)

# SDL3 Options
set(SDL_SHARED OFF CACHE BOOL "" FORCE)
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_TESTS OFF CACHE BOOL "" FORCE)
set(SDL_TEST OFF CACHE BOOL "" FORCE)
set(SDL_INSTALL OFF CACHE BOOL "" FORCE)

# Disable XScreenSaver dependency check prior to invoking add_subdirectory
set(SDL_X11_XSCREENSAVER OFF CACHE BOOL "" FORCE)
set(SDL_XSCREENSAVER OFF CACHE BOOL "" FORCE)
set(SDL_X11_XSCRNSAVER OFF CACHE BOOL "" FORCE)
set(SDL_WAYLAND OFF CACHE BOOL "" FORCE) # Added to skip Wayland XML generation

# -----------------------------------------------------------------------------
# 3. Add Subdirectories
# -----------------------------------------------------------------------------
add_subdirectory(3rdParty/ggml EXCLUDE_FROM_ALL)
add_subdirectory(3rdParty/SDL EXCLUDE_FROM_ALL)

# -----------------------------------------------------------------------------
# 4. Configure Dear ImGui Target (Vulkan + SDL3)
# -----------------------------------------------------------------------------
file(GLOB IMGUI_SOURCES
        3rdParty/imgui/*.cpp
        3rdParty/imgui/backends/imgui_impl_sdl3.cpp
        3rdParty/imgui/backends/imgui_impl_vulkan.cpp
)

add_library(imgui STATIC ${IMGUI_SOURCES})

target_include_directories(imgui PUBLIC
        3rdParty/imgui
        3rdParty/imgui/backends
)

# Link SDL3 and Vulkan to ImGui
target_link_libraries(imgui PUBLIC
        SDL3::SDL3-static
        Vulkan::Vulkan
)

# -----------------------------------------------------------------------------
# 5. Define Your Target and Link Dependencies
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME} TML_main.cc TML_window.h TML_memory.h TML_ggml.h)

target_link_libraries(${PROJECT_NAME} PRIVATE
        ggml
        SDL3::SDL3-static
        imgui
        Vulkan::Vulkan
)
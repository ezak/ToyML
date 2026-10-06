cmake_minimum_required(VERSION 3.16)
project(ToyML C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# -----------------------------------------------------------------------------
# Find Required System Packages
# -----------------------------------------------------------------------------
find_package(Vulkan REQUIRED)

# -----------------------------------------------------------------------------
# Configure Options BEFORE add_subdirectory
# -----------------------------------------------------------------------------
# GGML Options
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)  # Enable GGML Vulkan backend
set(GGML_VULKAN_SHADERS_GEN OFF CACHE BOOL "" FORCE)

# -----------------------------------------------------------------------------
# Add Subdirectories
# -----------------------------------------------------------------------------
add_subdirectory(3rdParty/ggml EXCLUDE_FROM_ALL)

# -----------------------------------------------------------------------------
# Define Your Target and Link Dependencies
# -----------------------------------------------------------------------------
foreach(_target bmt )
    add_executable(${PROJECT_NAME}-${_target} "${_target}.cc")
    target_link_libraries(${PROJECT_NAME}-${_target} ggml Vulkan::Vulkan)
endforeach()
cmake_minimum_required(VERSION 3.16)
project(ToyML C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# -----------------------------------------------------------------------------
# 1. Configure GGML Build Options BEFORE add_subdirectory
# -----------------------------------------------------------------------------
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Hardware acceleration options (enable as needed for your platform)
# set(GGML_CUDA ON CACHE BOOL "" FORCE)
# set(GGML_METAL ON CACHE BOOL "" FORCE)
# set(GGML_VULKAN ON CACHE BOOL "" FORCE)

# -----------------------------------------------------------------------------
# 2. Add GGML Subdirectory
# -----------------------------------------------------------------------------
add_subdirectory(3rdParty/ggml EXCLUDE_FROM_ALL)

# -----------------------------------------------------------------------------
# 3. Define Your Target and Link GGML
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME} main.cc TML_memory.h TML_ggml.h)

# Link the primary ggml target
target_link_libraries(${PROJECT_NAME} PRIVATE ggml)

# If you are using specific backends (CPU, CUDA, Metal), also link ggml-base or backend targets
# target_link_libraries(my_app PRIVATE ggml ggml-base)

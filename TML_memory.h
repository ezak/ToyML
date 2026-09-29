#ifndef TOYML_MEMORY_H
#define TOYML_MEMORY_H

#include <iostream>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

class TML_memory {
public:
    uint8_t *mapped_data = nullptr;
    size_t file_size = 0;
    uint64_t header_size = 0;

    TML_memory() = default;

    // Disable copying to prevent double-munmap/close bugs
    TML_memory(const TML_memory &) = delete;

    TML_memory &operator=(const TML_memory &) = delete;

    // Enable move semantics
    TML_memory(TML_memory &&other) noexcept
        : mapped_data(std::exchange(other.mapped_data, nullptr)),
          file_size(std::exchange(other.file_size, 0)),
          header_size(std::exchange(other.header_size, 0)) {
    }

    TML_memory &operator=(TML_memory &&other) noexcept {
        if (this != &other) {
            unmap();
            mapped_data = std::exchange(other.mapped_data, nullptr);
            file_size = std::exchange(other.file_size, 0);
            header_size = std::exchange(other.header_size, 0);
        }
        return *this;
    }

    ~TML_memory() {
        unmap();
    }

    void unmap() {
        if (mapped_data && mapped_data != MAP_FAILED) {
            ::munmap(mapped_data, file_size);
            mapped_data = nullptr;
        }
        file_size = 0;
        header_size = 0;
    }

    int map(const std::string &file_path) {
        unmap(); // Clean up previous allocation if re-mapped

        const int fd = ::open(file_path.c_str(), O_RDONLY);
        if (fd == -1) {
            std::cerr << "[TML_memory] Open failed: " << file_path << "\n";
            return 1;
        }

        struct stat st = {};
        if (::fstat(fd, &st) == -1) {
            std::cerr << "[TML_memory] Stat failed: " << file_path << "\n";
            ::close(fd);
            return 1;
        }

        if (st.st_size < static_cast<off_t>(sizeof(uint64_t))) {
            std::cerr << "[TML_memory] Invalid safetensors file size.\n";
            ::close(fd);
            return 1;
        }

        file_size = static_cast<size_t>(st.st_size);

        mapped_data = static_cast<uint8_t *>(
            ::mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0)
        );

        // Kernel mapping reference is established; descriptor can be safely closed immediately.
        ::close(fd);

        if (mapped_data == MAP_FAILED) {
            mapped_data = nullptr;
            std::cerr << "[TML_memory] mmap failed.\n";
            return 1;
        }

        // Prefetch hint for page tables
        ::madvise(mapped_data, file_size, MADV_WILLNEED);

        // Parse 8-byte header size (Little-Endian)
        std::memcpy(&header_size, mapped_data, sizeof(uint64_t));

        // Validate header size against file boundaries
        if (sizeof(uint64_t) + header_size > file_size) {
            std::cerr << "[TML_memory] Header size exceeds total file length.\n";
            unmap();
            return 1;
        }

        return 0;
    }

    [[nodiscard]] const uint8_t *json_data() const noexcept {
        return mapped_data ? mapped_data + sizeof(uint64_t) : nullptr;
    }

    [[nodiscard]] const uint8_t *tensor_data() const noexcept {
        return mapped_data ? mapped_data + sizeof(uint64_t) + header_size : nullptr;
    }
};

#endif // TOYML_MEMORY_H

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <spdlog/spdlog.h>
#include <vector>

// 解析ELF后返回的需要分配的内存块信息
typedef struct MemBlock {
    uint64_t vaddr;            // 内存块的起始地址
    size_t memsz;              // 内存块所需分配大小
    std::vector<uint8_t> data; // 此内存块的初始化数据
    // 若(filesz=)data.size < memsz则需要补0到memsz大小
    MemBlock() : vaddr(0), memsz(0), data() {}
    MemBlock(uint64_t vaddr_, size_t memsz_, const std::vector<uint8_t> &data_)
        : vaddr(vaddr_), memsz(memsz_), data(data_) {}
    MemBlock(uint64_t vaddr_, size_t memsz_)
        : vaddr(vaddr_), memsz(memsz_), data() {}
} MemBlock;

std::vector<MemBlock> get_data_from_elf(const char *filename, std::shared_ptr<spdlog::logger> logger);

// Look up a symbol value (address) from an ELF file. Returns std::nullopt if not found.
std::optional<uint64_t> get_symbol_value_from_elf(const char *filename,
                                                  const char *symbol_name,
                                                  std::shared_ptr<spdlog::logger> logger);

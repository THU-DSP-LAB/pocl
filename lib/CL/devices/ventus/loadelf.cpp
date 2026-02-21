#include "loadelf.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <fcntl.h>
#include <gelf.h>
#include <libelf.h>
#include <memory>
#include <spdlog/spdlog.h>
#include <unistd.h>

// 分析 ELF 文件，返回所有加载到内存中的段信息
std::vector<MemBlock> get_data_from_elf(const char *filename, std::shared_ptr<spdlog::logger> logger) {
    std::vector<MemBlock> blocks;
    if (!logger) {
        logger = spdlog::default_logger();
    }

    // 初始化 libelf 库
    if (elf_version(EV_CURRENT) == EV_NONE) {
        logger->error("ELF: cannot initialize libelf");
        return std::vector<MemBlock>();
    }

    // 打开 ELF 文件
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        const char *errstr = strerror(errno);
        logger->error("ELF: cannot open file '{}': {}", filename, errstr);
        return std::vector<MemBlock>();
    }

    // 开始 ELF 处理
    Elf *e = elf_begin(fd, ELF_C_READ, nullptr);
    if (!e) {
        logger->error("ELF: elf_begin failed: {}", elf_errmsg(-1));
        close(fd);
        return std::vector<MemBlock>();
    }

    // 读取 ELF 头部
    GElf_Ehdr ehdr;
    if (gelf_getehdr(e, &ehdr) == nullptr) {
        logger->error("ELF: gelf_getehdr failed: {}", elf_errmsg(-1));
        elf_end(e);
        close(fd);
        return std::vector<MemBlock>();
    }

    // 获取程序头表中的段数量
    size_t phnum = 0;
    if (elf_getphdrnum(e, &phnum) != 0) {
        logger->error("ELF: elf_getphdrnum failed: {}", elf_errmsg(-1));
        elf_end(e);
        close(fd);
        return std::vector<MemBlock>();
    }

    // 遍历所有程序头
    for (size_t i = 0; i < phnum; ++i) {
        GElf_Phdr phdr;
        if (gelf_getphdr(e, i, &phdr) != &phdr) {
            logger->error("ELF: gelf_getphdr header {} failed: {}", i, elf_errmsg(-1));
            return std::vector<MemBlock>();
        }

        // 只考虑加载段
        if (phdr.p_type == PT_LOAD) {
            MemBlock block;
            block.vaddr = phdr.p_vaddr;
            block.memsz = phdr.p_memsz;

            // 如果段在文件中有初始化数据，则读取数据
            if (phdr.p_filesz > 0) {
                block.data.resize(phdr.p_filesz);
                if (lseek(fd, phdr.p_offset, SEEK_SET) == -1) {
                    const char *errstr = strerror(errno);
                    logger->error("ELF: failed seeking to offset {}: {}", phdr.p_offset, errstr);
                    return std::vector<MemBlock>();
                } else {
                    ssize_t bytesRead = read(fd, block.data.data(), phdr.p_filesz);
                    if (bytesRead != (ssize_t)phdr.p_filesz) {
                        const char *errstr = strerror(errno);
                        logger->error(
                            "ELF: failed reading {} bytes data: {}", phdr.p_filesz, errstr
                        );
                        return std::vector<MemBlock>();
                    }
                }
            }
            blocks.push_back(block);
        }
    }

    elf_end(e);
    close(fd);
    return blocks;
}

std::optional<uint64_t> get_symbol_value_from_elf(const char *filename,
                                                  const char *symbol_name,
                                                  std::shared_ptr<spdlog::logger> logger) {
    if (!symbol_name || symbol_name[0] == '\0') {
        return std::nullopt;
    }
    if (!logger) {
        logger = spdlog::default_logger();
    }

    if (elf_version(EV_CURRENT) == EV_NONE) {
        logger->error("ELF: cannot initialize libelf");
        return std::nullopt;
    }

    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        const char *errstr = strerror(errno);
        logger->error("ELF: cannot open file '{}': {}", filename, errstr);
        return std::nullopt;
    }

    Elf *e = elf_begin(fd, ELF_C_READ, nullptr);
    if (!e) {
        logger->error("ELF: elf_begin failed: {}", elf_errmsg(-1));
        close(fd);
        return std::nullopt;
    }

    std::optional<uint64_t> result = std::nullopt;

    Elf_Scn *scn = nullptr;
    while ((scn = elf_nextscn(e, scn)) != nullptr) {
        GElf_Shdr shdr;
        if (gelf_getshdr(scn, &shdr) == nullptr) {
            logger->error("ELF: gelf_getshdr failed: {}", elf_errmsg(-1));
            break;
        }

        if (shdr.sh_type != SHT_SYMTAB && shdr.sh_type != SHT_DYNSYM) {
            continue;
        }

        Elf_Data *data = elf_getdata(scn, nullptr);
        if (!data || data->d_size == 0) {
            continue;
        }

        const size_t count = shdr.sh_entsize ? (shdr.sh_size / shdr.sh_entsize) : 0;
        for (size_t i = 0; i < count; ++i) {
            GElf_Sym sym;
            if (gelf_getsym(data, (int)i, &sym) == nullptr) {
                continue;
            }

            const char *name = elf_strptr(e, shdr.sh_link, sym.st_name);
            if (!name) {
                continue;
            }

            if (strcmp(name, symbol_name) != 0) {
                continue;
            }

            // Ignore undefined symbols.
            if (sym.st_shndx == SHN_UNDEF) {
                continue;
            }

            result = (uint64_t)sym.st_value;
            break;
        }

        if (result.has_value()) {
            break;
        }
    }

    elf_end(e);
    close(fd);
    return result;
}

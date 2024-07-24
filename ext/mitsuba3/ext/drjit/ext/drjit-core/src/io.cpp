/*
    src/io.cpp -- Disk cache for LLVM/CUDA kernels

    Copyright (c) 2021 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "io.h"
#include "log.h"
#include "internal.h"
#include "profiler.h"
#include "cuda_api.h"
#include "optix_api.h"
#include "../resources/kernels.h"
#include <stdexcept>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <lz4.h>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <unistd.h>
#  include <sys/mman.h>
#endif

/// Version number for cache files
#define DRJIT_CACHE_VERSION 4

// Uncomment to write out training data for creating a compression dictionary
// #define DRJIT_CACHE_TRAIN 1

#pragma pack(push)
#pragma pack(1)
struct CacheFileHeader {
    uint8_t version;
    uint32_t compressed_size;
    uint32_t source_size;
    uint32_t kernel_size;
    uint32_t reloc_size;
};
#pragma pack(pop)

char jitc_lz4_dict[jitc_lz4_dict_size];
static bool jitc_lz4_dict_ready = false;

void jitc_lz4_init() {
    if (jitc_lz4_dict_ready)
        return;

    if (jitc_lz4_dict_size != kernels_dict_size_uncompressed)
        jitc_fail("jit_init_lz4(): dictionary has invalid size!");

    if (LZ4_decompress_safe(kernels_dict, jitc_lz4_dict,
                            kernels_dict_size_compressed,
                            kernels_dict_size_uncompressed) !=
        (int) kernels_dict_size_uncompressed)
        jitc_fail("jit_init_lz4(): decompression of dictionary failed!");

    jitc_lz4_dict_ready = true;
}

bool jitc_kernel_load(const char *source, uint32_t source_size,
                      JitBackend backend, XXH128_hash_t hash, Kernel &kernel) {
    jitc_lz4_init();

#if !defined(_WIN32)
    char filename[512];
    if (unlikely(snprintf(filename, sizeof(filename), "%s/%016llx%016llx.%s.bin",
                          jitc_temp_path, (unsigned long long) hash.high64,
                          (unsigned long long) hash.low64,
                          backend == JitBackend::CUDA ? "cuda" : "llvm") < 0))
        jitc_fail("jit_kernel_load(): scratch space for filename insufficient!");

    int fd = open(filename, O_RDONLY);
    if (fd == -1)
        return false;

    auto read_retry = [&](uint8_t* data, size_t data_size) {
        while (data_size > 0) {
            ssize_t n_read = read(fd, data, data_size);
            if (n_read <= 0) {
                if (errno == EINTR) {
                    continue;
                } else {
                    jitc_raise("jit_kernel_load(): I/O error while while "
                              "reading compiled kernel from cache "
                              "file \"%s\": %s",
                              filename, strerror(errno));
                }
            }
            data += n_read;
            data_size -= n_read;
        }
    };
#else
    wchar_t filename_w[512];
    char filename[512];

    int rv = _snwprintf(filename_w, sizeof(filename_w) / sizeof(wchar_t),
                        L"%s\\%016llx%016llx.%s.bin",
                        jitc_temp_path, (unsigned long long) hash.high64,
                        (unsigned long long) hash.low64,
                        backend == JitBackend::CUDA ? L"cuda" : L"llvm");

    if (rv < 0 || rv == sizeof(filename) ||
        wcstombs(filename, filename_w, sizeof(filename)) == sizeof(filename))
        jitc_fail("jit_kernel_load(): scratch space for filename insufficient!");

    HANDLE fd = CreateFileW(filename_w, GENERIC_READ,
        FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);

    if (fd == INVALID_HANDLE_VALUE)
        return false;

    auto read_retry = [&](uint8_t* data, size_t data_size) {
        while (data_size > 0) {
            DWORD n_read = 0;
            if (!ReadFile(fd, data, (DWORD) data_size, &n_read, nullptr) || n_read == 0)
                jitc_raise("jit_kernel_load(): I/O error while while "
                           "reading compiled kernel from cache "
                           "file \"%s\": %u", filename, GetLastError());

            data += n_read;
            data_size -= n_read;
        }
    };
#endif

    char *compressed = nullptr, *uncompressed = nullptr;

    CacheFileHeader header;
    bool success = true;
    std::vector<void *> func;

    try {
        read_retry((uint8_t *) &header, sizeof(CacheFileHeader));

        if (header.version != DRJIT_CACHE_VERSION)
            jitc_raise("jit_kernel_load(): cache file \"%s\" is from an "
                       "incompatible version of Dr.Jit. You may want to wipe "
                       "your ~/.drjit directory.", filename);

        if (header.source_size != source_size)
            jitc_raise("jit_kernel_load(): cache collision in file \"%s\": size "
                       "mismatch (%u vs %u bytes).",
                       filename, header.source_size, source_size);

        uint32_t uncompressed_size =
            header.source_size + header.kernel_size + header.reloc_size;

        compressed = (char *) malloc_check(header.compressed_size);
        uncompressed = (char *) malloc_check(size_t(uncompressed_size) + jitc_lz4_dict_size);
        memcpy(uncompressed, jitc_lz4_dict, jitc_lz4_dict_size);

        read_retry((uint8_t *) compressed, header.compressed_size);

        uint32_t rv_2 = (uint32_t) LZ4_decompress_safe_usingDict(
            compressed, uncompressed + jitc_lz4_dict_size,
            (int) header.compressed_size, (int) uncompressed_size,
            (char *) uncompressed, jitc_lz4_dict_size);

        if (rv_2 != uncompressed_size)
            jitc_raise("jit_kernel_load(): cache file \"%s\" is malformed.",
                       filename);
    } catch (const std::exception &e) {
        jitc_log(Warn, "%s", e.what());
        success = false;
    }

    char *uncompressed_data = uncompressed + jitc_lz4_dict_size;

    if (success && memcmp(uncompressed_data, source, source_size) != 0) {
        jitc_log(Warn, "jit_kernel_load(): cache collision in file \"%s\".", filename);
        success = false;
    }

    if (success) {
        jitc_log(Trace, "jit_kernel_load(\"%s\")", filename);
        kernel.size = header.kernel_size;
        if (backend == JitBackend::CUDA) {
            kernel.data = malloc_check(header.kernel_size);
            memcpy(kernel.data, uncompressed_data + source_size, header.kernel_size);
        } else {
#if !defined(_WIN32)
            kernel.data = mmap(nullptr, header.kernel_size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (kernel.data == MAP_FAILED)
                jitc_fail("jit_llvm_load(): could not mmap() memory: %s",
                         strerror(errno));

            memcpy(kernel.data, uncompressed_data + source_size, header.kernel_size);
#else
            kernel.data = VirtualAlloc(nullptr, header.kernel_size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            if (!kernel.data)
                jitc_fail("jit_llvm_load(): could not VirtualAlloc() memory: %u", GetLastError());
            memcpy(kernel.data, uncompressed_data + source_size, header.kernel_size);

#endif
            uintptr_t *reloc = (uintptr_t *) (uncompressed_data + header.source_size + header.kernel_size);
            kernel.llvm.n_reloc = header.reloc_size / sizeof(void *);
            kernel.llvm.reloc = (void **) malloc(header.reloc_size);
            for (uint32_t i = 0; i < kernel.llvm.n_reloc; ++i)
                kernel.llvm.reloc[i] = (uint8_t *) kernel.data + reloc[i];

            // Write address of @vcall_table
            if (kernel.llvm.n_reloc > 1)
                *((void **) kernel.llvm.reloc[1]) = kernel.llvm.reloc + 1;

#if !defined(_WIN32)
            if (mprotect(kernel.data, header.kernel_size, PROT_READ | PROT_EXEC) == -1)
                jitc_fail("jit_llvm_load(): mprotect() failed: %s", strerror(errno));
#else
            DWORD unused;
            if (VirtualProtect(kernel.data, header.kernel_size, PAGE_EXECUTE_READ, &unused) == 0)
                jitc_fail("jit_llvm_load(): VirtualProtect() failed: %u", GetLastError());
#endif

#if defined(DRJIT_ENABLE_ITTNOTIFY)
            char name[39];
            snprintf(name, sizeof(name), "drjit_%016llx%016llx",
                     (unsigned long long) hash.high64,
                     (unsigned long long) hash.low64);
            kernel.llvm.itt = __itt_string_handle_create(name);
#endif
        }
    }

    free(compressed);
    free(uncompressed);

#if !defined(_WIN32)
    close(fd);
#else
    CloseHandle(fd);
#endif

    return success;
}

bool jitc_kernel_write(const char *source, uint32_t source_size,
                       JitBackend backend, XXH128_hash_t hash,
                       const Kernel &kernel) {
    jitc_lz4_init();

#if !defined(_WIN32)
    char filename[512], filename_tmp[512];
    if (unlikely(snprintf(filename, sizeof(filename), "%s/%016llx%016llx.%s.bin",
                          jitc_temp_path, (unsigned long long) hash.high64,
                          (unsigned long long) hash.low64,
                          backend == JitBackend::CUDA ? "cuda" : "llvm") < 0))
        jitc_fail("jit_kernel_write(): scratch space for filename insufficient!");

    if (unlikely(snprintf(filename_tmp, sizeof(filename_tmp), "%s.tmp",
                          filename) < 0))
        jitc_fail("jit_kernel_write(): scratch space for filename insufficient!");

    mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    int fd = open(filename_tmp, O_CREAT | O_EXCL | O_WRONLY, mode);
    if (fd == -1) {
        jitc_log(Warn,
                "jit_kernel_write(): could not write compiled kernel "
                "to cache file \"%s\": %s", filename_tmp, strerror(errno));
        return false;
    }

    auto write_retry = [&](const uint8_t *data, size_t data_size) {
        while (data_size > 0) {
            ssize_t n_written = write(fd, data, data_size);
            if (n_written <= 0) {
                if (errno == EINTR) {
                    continue;
                } else {
                    jitc_raise("jit_kernel_write(): I/O error while while "
                              "writing compiled kernel to cache "
                              "file \"%s\": %s",
                              filename_tmp, strerror(errno));
                }
            }
            data += n_written;
            data_size -= n_written;
        }
    };
#else
    wchar_t filename_w[512], filename_tmp_w[512];
    char filename[512], filename_tmp[512];

    int rv = _snwprintf(filename_w, sizeof(filename_w) / sizeof(wchar_t),
                        L"%s\\%016llx%016llx.%s.bin",
                        jitc_temp_path, (unsigned long long) hash.high64,
                        (unsigned long long) hash.low64,
                        backend == JitBackend::CUDA ? L"cuda" : L"llvm");

    if (rv < 0 || rv == sizeof(filename) ||
        wcstombs(filename, filename_w, sizeof(filename)) == sizeof(filename))
        jitc_fail("jit_kernel_write(): scratch space for filename insufficient!");

    rv = _snwprintf(filename_tmp_w, sizeof(filename_tmp_w) / sizeof(wchar_t),
                    L"%s.tmp", filename_w);

    if (rv < 0 || rv == sizeof(filename_tmp) ||
        wcstombs(filename_tmp, filename_tmp_w, sizeof(filename_tmp)) == sizeof(filename_tmp))
        jitc_fail("jit_kernel_write(): scratch space for filename insufficient!");

    HANDLE fd = CreateFileW(filename_tmp_w, GENERIC_WRITE,
        0 /* exclusive */, nullptr, CREATE_NEW /* fail if creation fails */,
        FILE_ATTRIBUTE_NORMAL, nullptr);

    if (fd == INVALID_HANDLE_VALUE) {
        jitc_log(Warn,
            "jit_kernel_write(): could not write compiled kernel "
            "to cache file \"%s\": %u", filename_tmp, GetLastError());
        return false;
    }

    auto write_retry = [&](const uint8_t* data, size_t data_size) {
        while (data_size > 0) {
            DWORD n_written = 0;
            if (!WriteFile(fd, data, (DWORD) data_size, &n_written, nullptr))
                jitc_raise("jit_kernel_write(): I/O error while while "
                          "writing compiled kernel to cache "
                          "file \"%s\": %u",
                          filename_tmp, GetLastError());

            data += n_written;
            data_size -= n_written;
        }
    };
#endif

    CacheFileHeader header;
    header.version = DRJIT_CACHE_VERSION;
    header.source_size = source_size;
    header.kernel_size = kernel.size;
    header.reloc_size = 0;

    if (backend == JitBackend::LLVM)
        header.reloc_size = kernel.llvm.n_reloc * sizeof(void *);

    uint32_t in_size =
                 header.source_size + header.kernel_size + header.reloc_size,
             out_size = LZ4_compressBound(in_size);

    uint8_t *temp_in  = (uint8_t *) malloc_check(in_size),
            *temp_out = (uint8_t *) malloc_check(out_size);

    memcpy(temp_in, source, header.source_size);
    memcpy(temp_in + source_size, kernel.data, header.kernel_size);

    if (backend == JitBackend::LLVM) {
        uintptr_t *reloc_out = (uintptr_t *) (temp_in + header.source_size + header.kernel_size);
        for (uint32_t i = 0; i < kernel.llvm.n_reloc; ++i)
            reloc_out[i] = (uintptr_t) kernel.llvm.reloc[i] - (uintptr_t) kernel.data;
    }

    LZ4_stream_t stream;
    memset(&stream, 0, sizeof(LZ4_stream_t));
    LZ4_resetStream_fast(&stream);
    LZ4_loadDict(&stream, jitc_lz4_dict, jitc_lz4_dict_size);

    header.compressed_size = (uint32_t) LZ4_compress_fast_continue(
        &stream, (const char *) temp_in, (char *) temp_out, (int) in_size,
        (int) out_size, 1);

    bool success = true;
    try {
        write_retry((const uint8_t *) &header, sizeof(CacheFileHeader));
        write_retry(temp_out, header.compressed_size);
    } catch (const std::exception &e) {
        jitc_log(Warn, "%s", e.what());
        success = false;
    }

    bool log = std::max(state.log_level_stderr,
                        state.log_level_callback) >= LogLevel::Trace;
    if (success && log)
        jitc_trace("jit_kernel_write(\"%s\"): compressed %s to %s", filename,
                  std::string(jitc_mem_string(size_t(source_size) + kernel.size)).c_str(),
                  std::string(jitc_mem_string(header.compressed_size)).c_str());

#if !defined(_WIN32)
    close(fd);

    if (link(filename_tmp, filename) != 0) {
        jitc_log(Warn,
            "jit_kernel_write(): could not link cache "
            "file \"%s\" into file system: %s",
            filename, strerror(errno));
        success = false;
    }

    if (unlink(filename_tmp) != 0) {
        jitc_raise("jit_kernel_write(): could not unlink temporary "
            "file \"%s\": %s",
            filename_tmp, strerror(errno));
        success = false;
    }
#else
    CloseHandle(fd);

    if (MoveFileW(filename_tmp_w, filename_w) == 0)
        jitc_log(Warn,
                "jit_kernel_write(): could not link cache "
                "file \"%s\" into file system: %u",
                filename, GetLastError());
#endif

#if DRJIT_CACHE_TRAIN == 1
    snprintf(filename, sizeof(filename), "%s/.drjit/%016llx%016llx.%s.trn",
             getenv("HOME"), (unsigned long long) hash.high64,
             (unsigned long long) hash.low64,
             backend == JitBackend::CUDA ? "cuda" : "llvm");
    fd = open(filename, O_CREAT | O_WRONLY, mode);
    if (fd) {
        write_retry(temp_in, in_size);
        close(fd);
    }
#endif

    free(temp_out);
    free(temp_in);

    return success;
}

void jitc_kernel_free(int device_id, const Kernel &kernel) {
    if (device_id == -1) {
#if !defined(_WIN32)
        if (munmap((void *) kernel.data, kernel.size) == -1)
            jitc_fail("jit_kernel_free(): munmap() failed!");
#else
        if (VirtualFree((void*) kernel.data, 0, MEM_RELEASE) == 0)
            jitc_fail("jit_kernel_free(): VirtualFree() failed!");
#endif
    } else {
        const Device &device = state.devices.at(device_id);
        if (kernel.size) {
            scoped_set_context guard(device.context);
            cuda_check(cuModuleUnload(kernel.cuda.mod));
            free(kernel.data);
        } else {
#if defined(DRJIT_ENABLE_OPTIX)
            jitc_optix_free(kernel);
#endif
        }
    }
}

void jitc_flush_kernel_cache() {
    jitc_log(Info, "jit_flush_kernel_cache(): releasing %zu kernel%s ..",
            state.kernel_cache.size(),
            state.kernel_cache.size() > 1 ? "s" : "");

    for (auto &v : state.kernel_cache) {
        jitc_kernel_free(v.first.device, v.second);
        free(v.first.str);
    }

    state.kernel_cache.clear();
}

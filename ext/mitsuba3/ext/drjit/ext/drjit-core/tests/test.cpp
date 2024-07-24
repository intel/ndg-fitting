#include "test.h"
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <algorithm>

#if defined(_WIN32)
#  include <windows.h>
#  include <direct.h>
#else
#  include <unistd.h>
#  include <sys/stat.h>
#endif

#if defined(__APPLE__)
#  include <mach-o/dyld.h>
#endif

#if defined(DRJIT_ENABLE_OPTIX)
#  include <drjit-core/optix.h>
#endif

struct Test {
    const char *name;
    void (*func)();
    const char *flags;
};

static std::vector<Test> *tests = nullptr;
std::string log_value;

extern "C" void log_level_callback(LogLevel, const char *msg) {
    log_value += msg;
    log_value += '\n';
}

/**
 * Strip away information (pointers, etc.) to that it becomes possible to
 * compare the debug output of a test to a reference file
 */
void test_sanitize_log(char *buf) {
    char *src = buf,
         *dst = src;

    // Remove all lines starting with the following text
    const char *excise[13] = {
        "jit_init(): detecting",
        " - Found CUDA",
        " - Enabling peer",
        "Detailed linker output:",
        "info    :",
        "ptxas",
        "     cache ",
        "jit_kernel_load(",
        "jit_kernel_write(",
        "jit_llvm_disasm()",
        "jit_llvm_mem_allocate(",
        "jit_kernel_read(): cache collision",
        "    // Grid-stride loop"
    };

    while (*src != '\0') {
        char c = *src;

        if (src == buf || src[-1] == '\n') {
            bool found = false;
            for (const char *str : excise) {
                if (strncmp(str, src, strlen(str)) == 0) {
                    while (*src != '\0' && src[0] != '\n')
                        src++;
                    if (*src == '\n' && *src != '\0')
                        src++;
                    found = true;
                    break;
                }
            }

            if (found)
                continue;
        }

        // Excise pointers, since they will differ from run to run
        if (c == '<' && src[1] == '0' && src[2] == 'x') {
            src += 3; c = *src;
            while ((c >= '0' && c <= '9') ||
                   (c >= 'a' && c <= 'f') ||
                   (c >= 'A' && c <= 'F') ||
                    c == '>')
                c = *++src;
            *dst++ = '<';
            *dst++ = '@';
            *dst++ = '>';
            continue;
        }

        /// Excise timing values
        if (strncmp(src, ", jit=", 6) == 0) {
            while (*src != ')')
                ++src;
        }

        if (strncmp(src, "-> launching ", 13) == 0 && src[13] != '<') {
            memcpy(dst, "-> launching <@>", 16);
            src += 29;
            dst += 16;
        }

        if (strncmp(src, "/* direct ptr */ 0x", 19) == 0) {
            while (*src != ';')
                src++;
            memcpy(dst, "<@>", 3);
            dst += 3;
            continue;
        }

        if (strncmp(src, "drjit_", 6) == 0 && src[6] != '<') {
            memcpy(dst, "drjit_<@>", 9);
            src += 38;
            dst += 9;
            continue;
        }

        if (strncmp(src, "func_", 5) == 0 && src[5] != '<') {
            memcpy(dst, "func_<@>", 8);
            src += 21;
            dst += 8;
            continue;
        }

        if (strncmp(src, "// sm_", 6) == 0) {
            src += 6 + 2;
            continue;
        }

        if (strncmp(src, "(reused local)", 14) == 0) {
            memcpy(dst, "(reused)", 8);
            src += 14;
            dst += 8;
            continue;
        }

        if (strncmp(src, "(reused global)", 15) == 0) {
            memcpy(dst, "(reused)", 8);
            src += 15;
            dst += 8;
            continue;
        }


        *dst++ = *src++;
    }
    *dst = '\0';
}

int test_check_log(const char *test_name, char *log, bool write_ref) {
    char test_fname[128];

    snprintf(test_fname, 128, "out_%s/%s.raw", TEST_NAME, test_name);

    FILE *f = fopen(test_fname, "w");
    if (!f) {
        fprintf(stderr, "\ntest_check_log(): Could not create file \"tests/%s\"!\n", test_fname);
        exit(EXIT_FAILURE);
    }

    size_t log_len = strlen(log);
    if (log_len > 0 && fwrite(log, log_len, 1, f) != 1) {
        fprintf(stderr, "\ntest_check_log(): Error writing to \"tests/%s\"!\n", test_fname);
        exit(EXIT_FAILURE);
    }
    fclose(f);

    test_sanitize_log(log);

    snprintf(test_fname, 128, "out_%s/%s.%s", TEST_NAME,
             test_name, write_ref ? "ref" : "out");

    f = fopen(test_fname, "w");
    if (!f) {
        fprintf(stderr, "\ntest_check_log(): Could not create file \"tests/%s\"!\n", test_fname);
        exit(EXIT_FAILURE);
    }

    log_len = strlen(log);
    if (log_len > 0 && fwrite(log, log_len, 1, f) != 1) {
        fprintf(stderr, "\ntest_check_log(): Error writing to \"tests/%s\"!\n", test_fname);
        exit(EXIT_FAILURE);
    }
    fclose(f);

    if (write_ref)
        return 1;

    snprintf(test_fname, 128, "out_%s/%s.ref", TEST_NAME, test_name);
    f = fopen(test_fname, "r");
    if (!f) {
        // fprintf(stderr, "\ntest_check_log(): Could not open file \"tests/%s\"!\n", test_fname);
        return 2;
    }

    int result = 1;

    fseek(f, 0, SEEK_END);
    result = (size_t) ftell(f) == log_len ? 1 : 0;
    fseek(f, 0, SEEK_SET);

    if (result) {
        char *tmp = (char *) malloc(log_len + 1);
        if (fread(tmp, log_len, 1, f) != 1) {
            fprintf(stderr, "\ntest_check_log(): Could not read file \"%s\"!\n", test_fname);
            fclose(f);
            return 0;
        }
        tmp[log_len] = '\0';

        test_sanitize_log(tmp);
        result = memcmp(tmp, log, log_len) == 0 ? 1 : 0;
        free(tmp);
    }

    fclose(f);
    return result;
}

int test_register(const char *name, void (*func)(), const char *flags) {
    if (!tests)
        tests = new std::vector<Test>();
    tests->push_back(Test{ name, func, flags });
    return 0;
}

int main(int argc, char **argv) {
#if defined(__APPLE__)
    char binary_path[1024];
    uint32_t binary_path_size = sizeof(binary_path);
    int rv = _NSGetExecutablePath(binary_path, &binary_path_size);
#elif defined(_WIN32)
    wchar_t binary_path[1024];
    int rv = GetModuleFileNameW(nullptr, binary_path, sizeof(binary_path) / sizeof(wchar_t));
#else
    char binary_path[1024];
    int rv = readlink("/proc/self/exe", binary_path, sizeof(binary_path) - 1);
    if (rv != -1)
        binary_path[rv] = '\0';
#endif

    if (rv < 0) {
        fprintf(stderr, "Unable to determine binary path!");
        exit(EXIT_FAILURE);
    }

#if !defined(_WIN32)
    char *last_slash = strrchr(binary_path, '/');
    if (!last_slash) {
        fprintf(stderr, "Invalid binary path!");
        exit(EXIT_FAILURE);
    }
    *last_slash='\0';
#else
    wchar_t* last_backslash = wcsrchr(binary_path, L'\\');
    if (!last_backslash) {
        fprintf(stderr, "Invalid binary path!");
        exit(EXIT_FAILURE);
    }
    *last_backslash = L'\0';
#endif

#if defined(_WIN32)
    rv = _wchdir(binary_path);
#else
    rv = chdir(binary_path);
#endif

    if (rv == -1) {
        fprintf(stderr, "Could not change working directory!");
        exit(EXIT_FAILURE);
    }

#if defined(_WIN32)
    _snwprintf(binary_path, 1024, L"out_%s", L"" TEST_NAME);
    _wmkdir(binary_path);
#else
    snprintf(binary_path, 1024, "out_%s", TEST_NAME);
    mkdir(binary_path, 0777);
#endif

    int log_level_stderr = (int) LogLevel::Warn;
    bool test_cuda = true, test_optix = true, test_llvm = true,
         write_ref = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-w") == 0) {
            write_ref = true;
        } else if (strcmp(argv[i], "-v") == 0) {
            log_level_stderr = std::max((int) LogLevel::Info, log_level_stderr + 1);
        } else if (strcmp(argv[i], "-c") == 0) {
            test_llvm = false;
            test_optix = false;
        } else if (strcmp(argv[i], "-l") == 0) {
            test_cuda = false;
            test_optix = false;
        } else if (strcmp(argv[i], "-o") == 0) {
            test_cuda = false;
            test_llvm = false;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            help = true;
        } else {
            fprintf(stderr, "Invalid command line argument: \"%s\"\n", argv[i]);
            help = true;
        }
    }

    if (help) {
        printf("Syntax: %s [options]\n\n", argv[0]);
        printf("Options:\n\n");
        printf(" -h   Display this help text.\n\n");
        printf(" -w   Write reference output to tests/out_*.\n\n");
        printf(" -e   Stop after the first failing test\n\n");
        printf(" -c   Only run CUDA tests\n\n");
        printf(" -l   Only run LLVM tests\n\n");
        printf(" -o   Only run OptiX tests\n\n");
        printf(" -v   Be more verbose (can be repeated)\n\n");
        return 0;
    }

    try {
        jit_set_log_level_stderr((LogLevel) log_level_stderr);
        jit_init((test_llvm ? (uint32_t) JitBackend::LLVM : 0) |
                 ((test_cuda || test_optix) ? (uint32_t) JitBackend::CUDA : 0));
        jit_set_log_level_callback(LogLevel::Trace, log_level_callback);
        fprintf(stdout, "\n");

        test_cuda &= (bool) jit_has_backend(JitBackend::CUDA);
        test_llvm &= (bool) jit_has_backend(JitBackend::LLVM);

#if defined(DRJIT_ENABLE_OPTIX)
        test_optix &= (bool) jit_has_backend(JitBackend::CUDA);
#else
        test_optix = false;
#endif

        if (!tests) {
            fprintf(stderr, "No tests registered!\n");
            exit(EXIT_FAILURE);
        }

        std::sort(tests->begin(), tests->end(), [](const Test &a, const Test &b) {
            return strcmp(a.name, b.name) < 0;
        });

        int tests_passed = 0,
            tests_failed = 0;

        for (auto &test : *tests) {
            fprintf(stdout, " - %s .. ", test.name);

            bool is_cuda = strstr(test.name, "_cuda"),
                 is_llvm = strstr(test.name, "_llvm"),
                 is_optix = strstr(test.name, "_optix");

            if ((is_cuda && !test_cuda) ||
                (is_optix && !test_optix) ||
                (is_llvm && !test_llvm)) {
                fprintf(stdout, "skipped.\n");
                continue;
            }

            fflush(stdout);
            log_value.clear();
            jit_init((uint32_t)((is_cuda || is_optix) ? JitBackend::CUDA
                                                      : JitBackend::LLVM));
#if defined(DRJIT_ENABLE_OPTIX)
            jit_set_flag(JitFlag::ForceOptiX, is_optix);
            if (is_optix)
                jit_optix_context();
#endif

#if !defined(__aarch64__)
            if (test_llvm)
                jit_llvm_set_target("skylake", nullptr, 8);
#endif

            auto before = std::chrono::high_resolution_clock::now();
            test.func();
            auto after = std::chrono::high_resolution_clock::now();
            float duration =
                std::chrono::duration_cast<
                    std::chrono::duration<float, std::milli>>(after - before)
                    .count();
            jit_shutdown(1);

            if (test_check_log(test.name, (char *) log_value.c_str(), write_ref) != 0) {
                tests_passed++;
                fprintf(stdout, "passed (%.2f ms).\n", duration);
            } else {
                tests_failed++;
                fprintf(stdout, "FAILED!\n");
            }
        }

        jit_shutdown(0);

        int tests_skipped = (int) tests->size() - tests_passed - tests_failed;
        if (tests_skipped == 0)
            fprintf(stdout, "\nPassed %i/%i tests.\n", tests_passed,
                    tests_passed + tests_failed);
        else
            fprintf(stdout, "\nPassed %i/%i tests (%i skipped).\n", tests_passed,
                    tests_passed + tests_failed, tests_skipped);

        return tests_failed == 0 ? 0 : EXIT_FAILURE;
    } catch (const std::exception &e) {
        fprintf(stderr, "Exception: %s!\n", e.what());
    }
}

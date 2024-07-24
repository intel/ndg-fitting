#pragma once

#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

extern const int                kernels_dict_size_uncompressed;
extern const int                kernels_dict_size_compressed;
extern const unsigned long long kernels_dict_hash_low64;
extern const unsigned long long kernels_dict_hash_high64;
extern const char               kernels_dict[];

extern const int                kernels_50_size_uncompressed;
extern const int                kernels_50_size_compressed;
extern const unsigned long long kernels_50_hash_low64;
extern const unsigned long long kernels_50_hash_high64;
extern const char               kernels_50[];

extern const int                kernels_70_size_uncompressed;
extern const int                kernels_70_size_compressed;
extern const unsigned long long kernels_70_hash_low64;
extern const unsigned long long kernels_70_hash_high64;
extern const char   kernels_70[];

extern const char   *kernels_list;

#if defined(__cplusplus)
}
#endif
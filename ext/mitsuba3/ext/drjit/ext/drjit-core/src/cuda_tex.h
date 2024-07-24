#include "cuda_api.h"

#if defined(DRJIT_DYNAMIC_CUDA)
using CUarray = struct CUarray_st *;
using CUtexObject = struct CUtexObject_st *;

#define CU_RESOURCE_TYPE_ARRAY 0
#define CU_TR_FILTER_MODE_POINT 0
#define CU_TR_FILTER_MODE_LINEAR 1
#define CU_TRSF_NORMALIZED_COORDINATES 2
#define CU_TR_ADDRESS_MODE_WRAP 0
#define CU_TR_ADDRESS_MODE_CLAMP 1
#define CU_TR_ADDRESS_MODE_MIRROR 2
#define CU_MEMORYTYPE_DEVICE 2
#define CU_MEMORYTYPE_ARRAY 3

#define CU_AD_FORMAT_FLOAT 0x20
#define CU_RES_VIEW_FORMAT_FLOAT_1X32 0x16
#define CU_RES_VIEW_FORMAT_FLOAT_2X32 0x17
#define CU_RES_VIEW_FORMAT_FLOAT_4X32 0x18

struct CUDA_ARRAY_DESCRIPTOR {
    size_t Width;
    size_t Height;
    int Format;
    unsigned int NumChannels;
};

struct CUDA_ARRAY3D_DESCRIPTOR {
    size_t Width;
    size_t Height;
    size_t Depth;
    int Format;
    unsigned int NumChannels;
    unsigned int Flags;
};

struct CUDA_RESOURCE_DESC {
    int resType;
    union {
        struct { CUarray hArray; } array;
        struct { int reserved[32]; } reserved;
    } res;
    unsigned int flags;
};

struct CUDA_TEXTURE_DESC {
    int addressMode[3];
    int filterMode;
    unsigned int flags;
    unsigned int maxAnisotropy;
    int mipmapFilterMode;
    float mipmapLevelBias;
    float minMipmapLevelClamp;
    float maxMipmapLevelClamp;
    float borderColor[4];
    int reserved[12];
};

struct CUDA_RESOURCE_VIEW_DESC {
    int format;
    size_t width;
    size_t height;
    size_t depth;
    unsigned int firstMipmapLevel;
    unsigned int lastMipmapLevel;
    unsigned int firstLayer;
    unsigned int lastLayer;
    unsigned int reserved[16];
};

struct CUDA_MEMCPY2D {
    size_t srcXInBytes;
    size_t srcY;
    int srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    int dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
};

struct CUDA_MEMCPY3D {
    size_t srcXInBytes;
    size_t srcY;
    size_t srcZ;
    size_t srcLOD;
    int srcMemoryType;
    const void *srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    void *reserved0;
    size_t srcPitch;
    size_t srcHeight;
    size_t dstXInBytes;
    size_t dstY;
    size_t dstZ;
    size_t dstLOD;
    int dstMemoryType;
    void *dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    void *reserved1;
    size_t dstPitch;
    size_t dstHeight;
    size_t WidthInBytes;
    size_t Height;
    size_t Depth;
};

extern CUresult (*cuArrayCreate)(CUarray *, const CUDA_ARRAY_DESCRIPTOR *);
extern CUresult (*cuArray3DCreate)(CUarray *, const CUDA_ARRAY3D_DESCRIPTOR *);
extern CUresult (*cuArray3DGetDescriptor)(CUDA_ARRAY3D_DESCRIPTOR *, CUarray);
extern CUresult (*cuArrayDestroy)(CUarray);
extern CUresult (*cuTexObjectCreate)(CUtexObject *, const CUDA_RESOURCE_DESC *,
                                     const CUDA_TEXTURE_DESC *,
                                     const CUDA_RESOURCE_VIEW_DESC *);
extern CUresult (*cuTexObjectDestroy)(CUtexObject);
extern CUresult (*cuTexObjectGetResourceDesc)(CUDA_RESOURCE_DESC *,
                                              CUtexObject);
extern CUresult (*cuMemcpy3DAsync)(const CUDA_MEMCPY3D *, CUstream);
extern CUresult (*cuMemcpy2DAsync)(const CUDA_MEMCPY2D *, CUstream);

#endif

extern void *jitc_cuda_tex_create(size_t ndim, const size_t *shape,
                                  size_t n_channels, int filter_mode,
                                  int wrap_mode);
extern void jitc_cuda_tex_get_shape(size_t ndim, const void *texture_handle,
                                    size_t *shape);
extern void jitc_cuda_tex_memcpy_d2t(size_t ndim, const size_t *shape,
                                     const void *src_ptr,
                                     void *dst_texture_handle);
extern void jitc_cuda_tex_memcpy_t2d(size_t ndim, const size_t *shape,
                                     const void *src_texture_handle,
                                     void *dst_ptr);
extern void jitc_cuda_tex_lookup(size_t ndim, const void *texture_handle,
                                 const uint32_t *pos, uint32_t *out);
extern void jitc_cuda_tex_bilerp_fetch(size_t ndim, const void *texture_handle,
                                       const uint32_t *pos, uint32_t *out);
extern void jitc_cuda_tex_destroy(void *texture_handle);

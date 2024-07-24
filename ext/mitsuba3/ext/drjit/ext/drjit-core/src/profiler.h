#if defined(DRJIT_ENABLE_ITTNOTIFY)
#  include <ittnotify.h>
#endif

#if defined(DRJIT_ENABLE_NVTX)
#  include <nvtx3/nvToolsExt.h>
#endif

#if defined(DRJIT_ENABLE_ITTNOTIFY)
extern __itt_domain *drjit_domain;
#endif

struct ProfilerRegion {
    ProfilerRegion(const char *name) : name(name) {
#if defined(DRJIT_ENABLE_ITTNOTIFY)
        itt_handle = __itt_string_handle_create(name);
#endif
    }

    const char *name;
#if defined(DRJIT_ENABLE_ITTNOTIFY)
    __itt_string_handle *itt_handle;
#endif
};

struct ProfilerPhase {
    ProfilerPhase(const ProfilerRegion &region) {
#if defined(DRJIT_ENABLE_ITTNOTIFY)
        __itt_task_begin(drjit_domain, __itt_null, __itt_null, region.itt_handle);
#endif
#if defined(DRJIT_ENABLE_NVTX)
        nvtxRangePush(region.name);
#endif
        (void) region;
    }
    ~ProfilerPhase() {
#if defined(DRJIT_ENABLE_ITTNOTIFY)
        __itt_task_end(drjit_domain);
#endif
#if defined(DRJIT_ENABLE_NVTX)
        nvtxRangePop();
#endif
    }
};

#include <stdint.h>

extern void jitc_vcall_set_self(JitBackend backend, uint32_t value, uint32_t index);
extern void jitc_vcall_self(JitBackend backend, uint32_t *value, uint32_t *index);

extern uint32_t jitc_var_loop_init(uint32_t *indices, uint32_t n_indices);

extern uint32_t jitc_var_vcall(const char *domain, uint32_t self, uint32_t mask,
                               uint32_t n_inst, const uint32_t *inst_id,
                               uint32_t n_in, const uint32_t *in,
                               uint32_t n_out_nested,
                               const uint32_t *out_nested,
                               const uint32_t *checkpoints, uint32_t *out);

extern VCallBucket *jitc_var_vcall_reduce(JitBackend backend,
                                          const char *domain, uint32_t index,
                                          uint32_t *bucket_count_out);

/// Helper data structure used to initialize the data block consumed by a vcall
struct VCallDataRecord {
    uint32_t offset;
    uint8_t size;
    bool literal;
    uint16_t unused;
    uintptr_t value;
};
static_assert(sizeof(VCallDataRecord) == 16, "VCallDataRecord has unexpected size!");

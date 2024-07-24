#include <stdint.h>

extern uint32_t jitc_var_loop_init(size_t n_indices, uint32_t **indices);

extern uint32_t jitc_var_loop_cond(uint32_t loop_var_init, uint32_t cond,
                                   size_t n_indices, uint32_t **indices);

extern uint32_t jitc_var_loop(const char *name, uint32_t loop_init,
                              uint32_t loop_cond, size_t n_indices,
                              uint32_t *indices_in, uint32_t **indices,
                              uint32_t checkpoint, int first_round);

extern void jitc_var_loop_simplify();

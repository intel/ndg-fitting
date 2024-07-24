<p align="center">
<img src="https://github.com/mitsuba-renderer/drjit-core/raw/master/resources/drjit-logo-dark.svg#gh-light-mode-only" alt="Dr.Jit logo" width="500"/>
<img src="https://github.com/mitsuba-renderer/drjit-core/raw/master/resources/drjit-logo-light.svg#gh-dark-mode-only" alt="Dr.Jit logo" width="500"/>
</p>

# Dr.Jit-Core — A tracing GPU/CPU Just-In-Time-Compiler

| Continuous Integration |
|         :---:          |
|   [![rgl-ci][1]][2]    |

[1]: https://rgl-ci.epfl.ch/app/rest/builds/aggregated/strob:(buildType:(project:(id:DrJitCore)))/statusIcon.svg
[2]: https://rgl-ci.epfl.ch/project/DrJitCore?mode=trends&guest=1


## Introduction

This repository contains an efficient and self-contained *just-in-time* (JIT)
compiler that can vectorize and parallelize computation. It was designed to to
accelerate differentiable Monte Carlo rendering that requires dynamic
compilation of large amounts of derivative code, though other types of
embarrassingly parallel computation are likely to benefit as well.

This library exposes a C and C++ interface that can be used to *trace*
computation, which means that the system internally builds a graph
representation of all steps while postponing their evaluation for as long as
possible. When the traced computation is finally evaluated, the system fuses
all operations into an efficient kernel containing queued computation that is
asynchronously evaluated on a desired device. On the CPU, this involves
compilation of vectorized [LLVM IR](https://llvm.org/docs/LangRef.html) and
parallel execution using a thread pool, while GPU compilation involves [NVIDIA
PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), and
either [CUDA](https://docs.nvidia.com/cuda) or
[OptiX](https://developer.nvidia.com/optix) depending on whether or not ray
tracing operations are used.

This project can be used independently or as part of the larger
[Dr.Jit](https://github.com/mitsuba-renderer/drjit) project, which furthermore
provides support for automatic differentiation, multidimensional
arrays/tensors, and a large library of mathematical functions.

The Dr.Jit-Core library has almost no dependencies: it can be compiled without
CUDA, OptiX, or LLVM actually being present on the system (it will attempt to
find them at runtime as needed). The library is implemented in C++14 but
exposes all functionality through a C99-compatible interface.

## Features

Dr.Jit has the following features:

- Runs on Linux (`X86_64`), macOS (`x86_64` & `aarch64`), and Windows
  (`x86_64`). Other platforms may work as well but have not been tested.

- Targets

    1. NVIDIA GPUs via CUDA (compute capability 5.0 or newer), and

    2. CPUs via LLVM leveraging available vector instruction set extensions
       (e.g. Neon or AVX/AVX2/AVX512).

- Captures and compiles pure arithmetic, side effects, and higher-level
  operations (loops and dynamic method dispatch) that are preserved 1:1 in
  generated kernels.

- Performs several basic optimizations to reduce the amount of LLVM/PTX IR
  passed to the next compiler stage.

  - Dead code elimination
  - Constant propagation
  - Common subexpression elimination via local value numbering

- Supports parallel kernel execution on multiple devices (JITing from several
  CPU threads, or running kernels on multiple GPUs).

- Provides a fast caching memory allocator that operates in the execution stream of
  an asynchronous computation device. This addresses a common performance bottleneck.

- Caches and reuses kernels when the same computation is encountered again.
  Caching is done both in memory and on disk (``~/.drjit`` on Linux and macOS,
  ``~/AppData/Local/Temp/drjit`` on Windows).

- Provides a variety of parallel reductions for convenience.

## An example (C++)

The header file
[drjit-core/array.h](https://github.com/mitsuba-renderer/drjit-core/blob/master/include/drjit-core/array.h)
provides a convenient C++ wrapper with operator operator overloading building
on the C-level API
([drjit-core/jit.h](https://github.com/mitsuba-renderer/drjit-core/blob/master/include/drjit-core/jit.h)).
Here is an brief example on how it can be used:

```cpp
#include <drjit-core/array.h>

using Bool   = CUDAArray<bool>;
using Float  = CUDAArray<float>;
using UInt32 = CUDAArray<uint32_t>;
```

The above snippet sets up a group of "capitalized" types that invoke the JIT
compiler. Any arithmetic involving instances of such types, e.g.,

```cpp
UInt32 c, a = /* .. */, b = /* .. */;

c = (a + b) * 5;
```

will conceptually expand to a parallel loop that processes the individual array
elements, e.g.,

```cpp
for (int i = 0; i < array_size; ++i) /* in parallel */ {
    uint32_t tmp0 = a[i] + b[i];
    c[i] = tmp0 * 5;
}
```

This evaluation of this loop is decoupled from the original program—in effect,
running the program _decides_ what the contents of this loop should be.

Let's look at a concrete example, using the previously defined types

```cpp
// Create a floating point array with 101 linearly spaced entries
// [0, 0.01, 0.02, ..., 1]
Float x = linspace<Float>(0, 1, 101);

// [0, 2, 4, 8, .., 98]
UInt32 index = arange<UInt32>(50) * 2;

// Equivalent to "y = x[index]"
Float y = gather(x, index);

/// Comparisons produce mask arrays
Bool mask = x < .5f;

// Ternary operator
Float z = select(mask, sqrt(x), 1.f / x);

printf("Value is = %s\n", z.str());
```

Running this program will trigger two kernel launches. The first generates the
``x`` array (size 100) when it is accessed by the ``gather()`` operation, and
the second generates ``z`` (size 50) when it is printed in the last line. Both
correspond to points during the execution where evaluation could no longer be
postponed, e.g., because of the cross-lane memory dependency in the former case.

Simply changing the first lines to

```cpp
#include <drjit-core/llvm.h>

using Bool   = LLVMArray<bool>;
using Float  = LLVMArray<float>;
using UInt32 = LLVMArray<uint32_t>;
```

switches to the functionally equivalent LLVM backend. By default, the LLVM
backend parallelizes execution via a built-in thread pool, enabling usage that
is very similar to the CUDA variant: a single thread issues computation that is
then processed in parallel by all cores of the system.

## How it works

To understand a bit better how all of this works, we can pop one level down to
the C-level interface. The first operation ``jit_init`` initializes Dr.Jit
and searches for LLVM and/or CUDA as instructed by the user. Note that users
don't need to install the CUDA SDK—just having an NVIDIA graphics driver is
enough.

```cpp
jit_init(JitBackendCUDA);
```

Let's calculate something: we will start by creating a single-precision
floating point variable that is initialized with the value ``0.5``. This
involves the function ``jit_var_new_stmt_0``, which creates a new variable that
depends on no other variables.

```cpp
uint32_t v0 = jit_var_new_stmt_0(/* backend = */ JitBackendCUDA,
                                 /* type    = */ VarTypeFloat32,
                                 /* stmt    = */ "mov.$t0 $r0, 0.5");
```
Note weird-looking code fragment ``mov.$t0 $r0, 0.5``. This is a *template* for
an operation expressed in
[PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
Familiarity with PTX is not needed to use Dr.Jit, and there are higher-level
wrappers for any conceivable standard operation. We simply show the lowest-level
interface here to illustrate how things fit together.

The ``$`` expressions are placeholders that Dr.Jit will replace with
something meaningful when the final program is generated. For example ``$t0``
is the type of the output argument (we could also have written ``f32`` as the
type is known here), and ``$r0`` is the register name associated to the output
argument. This is a *scalar* variable, which means that it will produce a
single element if evaluated alone, but it can also occur in any computation
involving larger arrays and will expand to the needed size.

Programs using Dr.Jit will normally create and destroy *vast* numbers of
variables, and this operation is therefore highly optimized. For example, the
default way of creating statements assumes that the instruction string template
exists in the program's data segment and therefore doesn't need to be copied.
The operation then creates an entry in a [very efficient hash
table](https://github.com/Tessil/robin-map) mapping the resulting variable
index ``v0`` to a record ``(backend, type, stmt, <operands>)``. Over time, this
hash table will expand to the size that is needed to support the active
computation, and from this point onward ``jit_var_new_..`` will not involve any
further dynamic memory allocation.

Let's do some computation with this variable: we can create a "counter", which
is an Dr.Jit array containing an increasing sequence of integer elements ``[0,
1, 2, .., 9]`` in this case.

```cpp
uint32_t v1 = jit_var_new_counter(/* backend = */ JitBackendCUDA,
                                  /* size    = */ 10);
```
Counters always have the variable type ``VarTypeUInt32`` that we next
convert into a single precision floating point variable.

```cpp
uint32_t v2 = jit_var_new_cast(/* index       = */ v1,
                               /* target_type = */ VarTypeFloat32,
                               /* reinterpret = */ 0);
```

Finally, let's create a more interesting variable that references some of the
previous results via ``op0`` and ``op1``.

```cpp
uint32_t v3 = jit_var_new_op_2(JitOpAdd, v0, v2)
```

Or, equivalently, using the lower-level string template interface:

```cpp
uint32_t v3 = jit_var_new_stmt_2(/* backend = */ JitBackendCUDA,
                                 /* type    = */ VarTypeFloat32,
                                 /* stmt    = */ "add.$t0 $r0, $r1, $r2",
                                 /* op0     = */ v0,
                                 /* op1     = */ v2);
```
Suppose that we don't plan to perform any
further computation / accesses involving ``v0``, ``v1``, and ``v2``. This must
be indicated to Dr.Jit by reducing their *external* reference count
("external" refers to references by *your* code):

```cpp
jit_var_dec_ref_ext(v0);
jit_var_dec_ref_ext(v1);
jit_var_dec_ref_ext(v2);
```

They still have a nonzero *internal* reference count (i.e. by Dr.Jit itself)
since the variable ``v3`` depends on them, and this keeps them from being
garbage-collected.

Note that no real computation has happened yet—so far, we were simply
manipulating hash table entries. Let's finally observe the result of this
calculation by printing the array contents:

```cpp
printf("Result: %s\n", jit_var_str(v3));
```

This step internally invokes ``jit_var_eval(v3)`` to evaluate the variable,
which creates a CUDA kernel containing all steps that are needed to compute
the contents of ``v3`` and write them into device-resident memory.

During this compilation step, the following happens: Dr.Jit first traverses
the relevant parts of the variable hash table and concatenates all string
templates (with appropriate substitutions) into a complete PTX representation.
This step is highly optimized and takes on the order of a few microseconds.

Once the final PTX string is available, two things can happen: potentially
we've never seen this particular sequence of steps before, and in that case the
PTX code must be further compiled to machine code ("SASS", or *streaming
assembly*). This step involves a full optimizing compiler embedded in the GPU
driver, which tends to be very slow: usually it's a factor of 1000-10000×
slower than the preceding steps within Dr.Jit.

However, once a kernel has been compiled, Dr.Jit will *remember* it using
both an in-memory and an on-disk cache. In programs that perform the same
sequence of steps over and over again (e.g. optimization), the slow PTX→SASS
compilation step will only occur in the first iteration. Evaluation of ``v2``
will turn the variable from a symbolic representation into a GPU-backed array,
and further queued computation accessing it will simply index into that array
instead of repeating the original computation.

At the end of the program, we must not forget to decrease the external
reference count associated with ``v2``, which will release the array from
memory. Finally, ``jit_shutdown()`` releases any remaining resources held by
Dr.Jit.

```cpp
jit_var_dec_ref_ext(v3);
jit_shutdown(0);
```

Running this program on a Linux machine provides the following output:

```
jit_init(): creating directory "/home/wjakob/.drjit" ..
jit_init(): detecting devices ..
jit_cuda_init(): enabling CUDA backend (version 11.1)
 - Found CUDA device 0: "GeForce RTX 3090" (PCI ID 65:00.0, compute cap. 8.6, 82 SMs w/99 KiB shared mem., 23.7 GiB global mem.)
jit_eval(): launching 1 kernel.
  -> launching e93e70f12fcaea9c (n=10, in=0, out=1, ops=8, jit=2.4 us):
     cache miss, build: 33.417 ms, 2.98 KiB.
jit_eval(): done.
Result: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
jit_shutdown(): releasing 1 kernel ..
jit_shutdown(): releasing 1 thread state ..
jit_shutdown(): done
jit_cuda_shutdown()
```

These log messages show that Dr.Jit generated a single kernel within 2.4 μs.
However, this kernel was never observed before, necessitating a compilation
step by the CUDA driver, which took 33 ms.

Note the ``Result: [...]`` line, which is the expected output of the
calculation ``0.5 + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]``. The extra lines are debug
statements that can be controlled by setting the log level to a higher or lower
level. A log callback can also be provided to e.g. route such messages to a
file.

Let's actually increase the log level to see some more detail of what is
happening under the hood. This can be done by adding the following two lines at
the beginning of the program

```cpp
jit_set_log_level_stderr(LogLevelDebug);
jit_set_flag(JitFlagPrintIR, 1);
```

This produces the following detailed output (there is also ``LogLevelTrace``
for the truly adventurous):

```
jit_init(): detecting devices ..
jit_cuda_init(): enabling CUDA backend (version 11.1)
 - Found CUDA device 0: "GeForce RTX 3090" (PCI ID 65:00.0, compute cap. 8.6, 82 SMs w/99 KiB shared mem., 23.7 GiB global mem.)
jit_var_new(float32 r1): mov.$t0 $r0, 0.5
jit_var_new(uint32 r2[10]): mov.u32 $r0, %r0
jit_var_new(float32 r3[10] <- r2): cvt.rn.$t0.$t1 $r0, $r1
jit_var_new_cast(float32 r3 <- uint32 r2)
jit_var_new(float32 r4[10] <- r1, r3): add.$t0 $r0, $r1, $r2
jit_eval(): launching 1 kernel.
  -> launching e93e70f12fcaea9c (n=10, in=0, out=1, ops=8, jit=2.9 us):
jit_eval(): launching 1 kernel.
.version 6.0
.target sm_60
.address_size 64

.entry drjit_e93e70f12fcaea9cecd06e2b4b9ab180(.param .align 8 .b8 params[16]) {
    .reg.b8   %b <8>; .reg.b16 %w<8>; .reg.b32 %r<8>;
    .reg.b64  %rd<8>; .reg.f32 %f<8>; .reg.f64 %d<8>;
    .reg.pred %p <8>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r0, %r0, %r1, %r2;
    ld.param.u32 %r2, [params];
    setp.ge.u32 %p0, %r0, %r2;
    @%p0 bra done;

    mov.u32 %r3, %nctaid.x;
    mul.lo.u32 %r1, %r3, %r1;

body: // sm_75
    mov.f32 %f4, 0.5;
    mov.u32 %r5, %r0;
    cvt.rn.f32.u32 %f6, %r5;
    add.f32 %f7, %f4, %f6;
    ld.param.u64 %rd0, [params+8];
    mad.wide.u32 %rd0, %r0, 4, %rd0;
    st.global.cs.f32 [%rd0], %f7;

    add.u32 %r0, %r0, %r1;
    setp.ge.u32 %p0, %r0, %r2;
    @!%p0 bra body;

done:
    ret;
}
     cache hit, load: 69.195 us, 2.98 KiB.
jit_eval(): cleaning up..
jit_eval(): done.
jit_shutdown(): releasing 1 kernel ..
jit_shutdown(): releasing 1 thread state ..
jit_flush_malloc_cache(): freed
 - device memory: 64 B in 1 allocation
jit_shutdown(): done
jit_cuda_shutdown()
Result: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
```

Note in particular the PTX fragments that includes the lines

```
L1: // Loop body
    mov.f32 %f4, 0.5;
    mov.u32 %r5, %r0;
    cvt.rn.f32.u32 %f6, %r5;
    add.f32 %f7, %f4, %f6;
```

These lines exactly corresponding to the variables ``v0`` to ``v3``
that we had previously defined. The surrounding code establishes a [grid-stride
loop](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)
that processes all array elements. This time around, the kernel compilation was
skipped, and Dr.Jit loaded the kernel from the on-disk cache file
``~/.drjit/e93e70f12fcaea9cecd06e2b4b9ab180.cuda.bin`` containing a
[LZ4](https://github.com/lz4/lz4)-compressed version of code and compilation
output. The odd hexadecimal value is simply the
[XXH3](https://cyan4973.github.io/xxHash/) hash of the kernel source code.

When a kernel includes OptiX function calls (ray tracing operations), kernels
are automatically launched through OptiX instead of the CUDA driver API.

### LLVM backend

The preceding section provided a basic example of Dr.Jit in combination with CUDA.
LLVM works essentially the same way. Now, the ``backend=`` flag must be set to
``JitBackendLLVM``, and ``stmt`` is expected to be a statement using LLVM's
textual intermediate representation. For example, the previous addition
operation would be written as

```cpp
uint32_t v3 = jit_var_new_2(/* backend = */ JitBackendLLVM,
                            /* type    = */ VarTypeFloat32,
                            /* stmt    = */ "$r0 = fadd <$w x $t0> $r1, $r2",
                            /* op0     = */ v0,
                            /* op1     = */ v2);
```

The LLVM backend operates on vectors matching the SIMD instruction set of the
host processor. For example, ``<$w x $t0>`` will e.g. expand to ``<8 x float>``
on a machine supporting the AVX/AVX2 instruction set, and ``<16 x float>`` on a
machine with AVX512.

A kernel transforming less than a few thousands of elements will be
JIT-compiled and executed immediately on the current thread. For large arrays,
Dr.Jit will automatically parallelize evaluation via a thread pool. The
repository includes
[nanothread](https://github.com/mitsuba-renderer/nanothread) as a git
submodule, which is a minimal implementation of the components that are
necessary to realize this. The size of this thread pool can also be set to
zero, in which case all computation will occur on the current thread. In this
case, another type of parallelism is available by using Dr.Jit from multiple
threads at once.

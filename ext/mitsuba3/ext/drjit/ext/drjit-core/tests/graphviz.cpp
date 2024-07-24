#include "test.h"
#include <cstring>

TEST_CUDA(01_graphviz) {
    Float r = linspace<Float>(0, 1, 11);
    jit_var_set_label(r.index(), "r");
    jit_prefix_push(Backend, "Scope 1");
    Float a = r + 1;
    jit_var_set_label(a.index(), "a");
    jit_prefix_pop(Backend);

    jit_prefix_push(Backend, "Scope 2");
    Float b = r + 2;
    jit_var_set_label(b.index(), "b");
    jit_prefix_push(Backend, "Nested scope");
    Float c = b + 3;
    jit_var_set_label(c.index(), "c");
    Float d = a + 4;
    jit_prefix_pop(Backend);
    jit_prefix_pop(Backend);
    Float e = r + 5;
    jit_var_set_label(e.index(), "e");

    jit_prefix_push(Backend, "Scope 2");
    jit_prefix_push(Backend, "Nested scope");
    Float f = a + 6;
    jit_prefix_pop(Backend);
    jit_prefix_pop(Backend);
    Float g = Float::steal(jit_var_wrap_vcall(f.index()));

    scatter_reduce(ReduceOp::Add, f, Float(4), UInt32(0));

    char *str = strdup(jit_var_graphviz());
    char *p = strstr(str, "Literal constant: 0x");
    jit_assert(p);
    p += 20;
    char *p2 = strstr(p, "|");
    jit_assert(p2);
    memset(p, '0', p2-p);

    const char *ref = R"(digraph {
    rankdir=TB;
    graph [dpi=50 fontname=Consolas];
    node [shape=record fontname=Consolas];
    edge [fontname=Consolas];
    1 [label="{mov.u32 $r0, %r0\l|{Type: cuda u32|Size: 11}|{ID #1|E:0|I:1}}}"];
    2 [label="{cvt.rn.$t0.$t1 $r0, $r1\l|{Type: cuda f32|Size: 11}|{ID #2|E:0|I:1}}}"];
    3 [label="{Literal constant: 0.1|{Type: cuda f32|Size: 1}|{ID #3|E:0|I:1}}}" fillcolor=gray90 style=filled];
    5 [label="{Label: \"r\"|mul.ftz.$t0 $r0, $r1, $r2\l|{Type: cuda f32|Size: 11}|{ID #5|E:1|I:3}}}" fillcolor=wheat style=filled];
    subgraph cluster_5615eefd04289ffb {
        label="Scope 1";
        6 [label="{Literal constant: 1|{Type: cuda f32|Size: 1}|{ID #6|E:0|I:1}}}" fillcolor=gray90 style=filled];
        7 [label="{Label: \"a\"|add.ftz.$t0 $r0, $r1, $r2\l|{Type: cuda f32|Size: 11}|{ID #7|E:1|I:1}}}" fillcolor=wheat style=filled];
    }
    subgraph cluster_6e8749cac8a1b5f3 {
        label="Scope 2";
        8 [label="{Literal constant: 2|{Type: cuda f32|Size: 1}|{ID #8|E:0|I:1}}}" fillcolor=gray90 style=filled];
        9 [label="{Label: \"b\"|add.ftz.$t0 $r0, $r1, $r2\l|{Type: cuda f32|Size: 11}|{ID #9|E:1|I:1}}}" fillcolor=wheat style=filled];
    }
    subgraph cluster_6e8749cac8a1b5f3 {
        label="Scope 2";
        subgraph cluster_2d27caeba104ea91 {
            label="Nested scope";
            10 [label="{Literal constant: 3|{Type: cuda f32|Size: 1}|{ID #10|E:0|I:1}}}" fillcolor=gray90 style=filled];
            11 [label="{Label: \"c\"|add.ftz.$t0 $r0, $r1, $r2\l|{Type: cuda f32|Size: 11}|{ID #11|E:1|I:0}}}" fillcolor=wheat style=filled];
            12 [label="{Literal constant: 4|{Type: cuda f32|Size: 1}|{ID #12|E:0|I:2}}}" fillcolor=gray90 style=filled];
            13 [label="{add.ftz.$t0 $r0, $r1, $r2\l|{Type: cuda f32|Size: 11}|{ID #13|E:1|I:0}}}"];
        }
    }
    14 [label="{Literal constant: 5|{Type: cuda f32|Size: 1}|{ID #14|E:0|I:1}}}" fillcolor=gray90 style=filled];
    15 [label="{Label: \"e\"|add.ftz.$t0 $r0, $r1, $r2\l|{Type: cuda f32|Size: 11}|{ID #15|E:1|I:0}}}" fillcolor=wheat style=filled];
    subgraph cluster_6e8749cac8a1b5f3 {
        label="Scope 2";
        subgraph cluster_2d27caeba104ea91 {
            label="Nested scope";
            17 [label="{Evaluated|{Type: cuda f32|Size: 11}|{ID #17|E:0|I:1}}}" fillcolor=lightblue2 style=filled];
        }
    }
    18 [label="{Placeholder|{Type: cuda f32|Size: 11}|{ID #18|E:1|I:0}}}" fillcolor=yellow style=filled];
    19 [label="{Literal constant: 0|{Type: cuda u32|Size: 1}|{ID #19|E:0|I:1}}}" fillcolor=gray90 style=filled];
    22 [label="{Evaluated (dirty)|{Type: cuda f32|Size: 11}|{ID #22|E:1|I:1}}}" fillcolor=salmon style=filled];
    23 [label="{Literal constant: 0x000000000000|{Type: cuda ptr|Size: 1}|{ID #23|E:0|I:1}}}" fillcolor=gray90 style=filled];
    24 [label="{mad.wide.$t3 %rd3, $r3, $s2, $r1\l.reg.$t2 $r0_unused\latom.global.add.$t2 $r0_unused, [%rd3], $r2\l|{Type: cuda void |Size: 1}|{ID #24|E:1|I:0}}}" fillcolor=yellowgreen style=filled];
    1 -> 2;
    2 -> 5 [label=" 1"];
    3 -> 5 [label=" 2"];
    5 -> 7 [label=" 1"];
    6 -> 7 [label=" 2"];
    5 -> 9 [label=" 1"];
    8 -> 9 [label=" 2"];
    9 -> 11 [label=" 1"];
    10 -> 11 [label=" 2"];
    7 -> 13 [label=" 1"];
    12 -> 13 [label=" 2"];
    5 -> 15 [label=" 1"];
    14 -> 15 [label=" 2"];
    17 -> 18;
    22 -> 23 [label=" 4"];
    23 -> 24 [label=" 1"];
    12 -> 24 [label=" 2"];
    19 -> 24 [label=" 3"];
    subgraph cluster_legend {
        label="Legend";
        l5 [style=filled fillcolor=yellow label="Placeholder"];
        l4 [style=filled fillcolor=yellowgreen label="Special"];
        l3 [style=filled fillcolor=salmon label="Dirty"];
        l2 [style=filled fillcolor=lightblue2 label="Evaluated"];
        l1 [style=filled fillcolor=wheat label="Labeled"];
        l0 [style=filled fillcolor=gray90 label="Constant"];
    }
}
)";
    FILE *f1 = fopen("a", "wb");
    FILE *f2 = fopen("b", "wb");
    fputs(ref, f1);
    fputs(str, f2);
    fclose(f1);
    fclose(f2);
    jit_assert(strcmp(ref, str) == 0);
}

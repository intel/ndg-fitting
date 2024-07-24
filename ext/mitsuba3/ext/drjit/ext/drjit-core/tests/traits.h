#pragma once

namespace drjit {
    namespace detail {
        /// Detector pattern that is used to drive many type traits below
        template <typename SFINAE, template <typename> typename Op, typename Arg>
        struct detector : std::false_type { };

        template <template <typename> typename Op, typename Arg>
        struct detector<std::void_t<Op<Arg>>, Op, Arg>
            : std::true_type { };

        template <typename T> using is_array_det = std::enable_if_t<T::IsArray>;
    };

    template <template<typename> class Op, typename Arg>
    constexpr bool is_detected_v = detail::detector<void, Op, Arg>::value;

    template <typename T>
    constexpr bool is_array_v = is_detected_v<detail::is_array_det, std::decay_t<T>>;

    template <typename T> using enable_if_jit_array_t = enable_if_t<is_array_v<T>>;
    template <typename T> using detached_t = T;
    template <typename T> constexpr size_t array_depth_v = is_array_v<T> ? 1 : 0;
    template <typename T> constexpr bool is_jit_array_v = is_array_v<T>;
    template <typename T> constexpr bool is_diff_array_v = false;
    template <typename T> constexpr bool is_drjit_struct_v = false;
    template <typename T> using mask_t = typename T::Mask;
    template <typename T> using scalar_t = typename T::Value;

    template <typename T> struct struct_support_t { };

    template <typename T> T detach(const T &value) { return value; }

    template <typename T> constexpr JitBackend backend_v = T::Backend;
}

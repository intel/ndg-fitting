#pragma once
#include <stdint.h>
#include <string>
#include <type_traits>
#include <thread>

#define INTEL_ITTNOTIFY_API_PRIVATE
#include "ittnotify.h"

namespace itt_notify {

template<bool bRegion = true>
class Task
{
protected:
    __itt_id m_id = __itt_null;
    const __itt_domain* m_pDomain;
public:
    Task(const __itt_domain* pDomain, __itt_string_handle* pName)
        : m_pDomain(pDomain)
    {
        m_id = __itt_id_make(const_cast<__itt_domain*>(m_pDomain), reinterpret_cast<unsigned long long>(pName));
        if (bRegion)
        {
            __itt_region_begin(m_pDomain, m_id, __itt_null, pName);
        }
        else
        {
            __itt_task_begin(m_pDomain, m_id, __itt_null, pName);
        }
    }

    template<class T>
    typename std::enable_if<std::is_floating_point<T>::value, void>::type AddArg(__itt_string_handle* pName, const T& value)
    {
        double double_value = value;
        __itt_metadata_add(m_pDomain, m_id, pName, __itt_metadata_double, 1, &double_value);
    }

    void AddArg(__itt_string_handle* pName, int64_t value)
    {
        __itt_metadata_add(m_pDomain, m_id, pName, __itt_metadata_s64, 1, &value);
    }

    void AddArg(__itt_string_handle* pName, const char* value)
    {
        __itt_metadata_str_add(m_pDomain, m_id, pName, value, 0);
    }

    void AddArg(__itt_string_handle* pName, void const* const pValue)
    {
        __itt_metadata_add(m_pDomain, m_id, pName, __itt_metadata_unknown, 1, const_cast<void*>(pValue));
    }

    ~Task()
    {
        if (bRegion)
        {
            __itt_region_end(m_pDomain, m_id);
        }
        else
        {
            __itt_task_end(m_pDomain);
        }
    }
};

#ifdef _WIN32
    #define UNICODE_AGNOSTIC(name) name##A
#else
    #define UNICODE_AGNOSTIC(name) name
#endif

#define ITT_DOMAIN(/*const char* */domain)\
    static const __itt_domain* __itt_domain_name = UNICODE_AGNOSTIC(__itt_domain_create)(domain)

#if defined(_MSC_VER) && _MSC_VER >= 1900 //since VS 2015 magic statics are supported, TODO: check with other compilers
    #define ITT_MAGIC_STATIC(static_variable)
#else
//the 'while' below is to protect code from crash in multi-threaded environment under compiler without magic statics support
    #define ITT_MAGIC_STATIC(static_variable) while(!(static_variable)) std::this_thread::yield();
#endif

#define ITT_SCOPE(region, name)\
    static __itt_string_handle* __itt_scope_name = UNICODE_AGNOSTIC(__itt_string_handle_create)(name);\
    ITT_MAGIC_STATIC(__itt_scope_name);\
    itt_notify::Task<region> __itt_scope_item(__itt_domain_name, __itt_scope_name)

#define ITT_SCOPE_TASK(/*const char* */name) ITT_SCOPE(false, name)
#define ITT_SCOPE_REGION(/*const char* */name) ITT_SCOPE(true, name)

#define ITT_FUNCTION_TASK() ITT_SCOPE_TASK(__FUNCTION__); ITT_ARG("__file__", __FILE__); ITT_ARG("__line__", __LINE__)

#define ITT_ARG(/*const char* */name, /*number or string*/ value) {\
    static __itt_string_handle* __itt_arg_name = UNICODE_AGNOSTIC(__itt_string_handle_create)(name);\
    ITT_MAGIC_STATIC(__itt_arg_name);\
    __itt_scope_item.AddArg(__itt_arg_name, value);\
}

enum Scope
{
    scope_global = __itt_scope_global,
    scope_process = __itt_scope_track_group,
    scope_thread =__itt_scope_track,
    scope_task =__itt_scope_task, //means a task that will long until another marker with task scope in this thread occurs
};

#define ITT_MARKER(/*const char* */name, /*enum Scope*/scope) {\
    static __itt_string_handle* __itt_marker_name = UNICODE_AGNOSTIC(__itt_string_handle_create)(name);\
    ITT_MAGIC_STATIC(__itt_marker_name);\
    __itt_marker(__itt_domain_name, __itt_null, __itt_marker_name, (__itt_scope)itt_notify::scope);\
}

#define ITT_COUNTER(/*const char* */name, /*double */value) { \
    static __itt_string_handle* __itt_counter_name = UNICODE_AGNOSTIC(__itt_string_handle_create)(name);\
    ITT_MAGIC_STATIC(__itt_counter_name);\
    double counter_value = value;\
    __itt_metadata_add(__itt_domain_name, __itt_null, __itt_counter_name, __itt_metadata_double, 1, &counter_value);\
}

class ScopeTrack
{
public:
    ScopeTrack(__itt_track* track)
    {
        __itt_set_track(track);
    }
    ~ScopeTrack()
    {
        __itt_set_track(nullptr);
    }
};

//'group' defines virtual process (null means current process), track defines virtual thread
#define ITT_SCOPE_TRACK(/*const char* */group, /*const char* */ track)\
    static __itt_track* itt_track_name = __itt_track_create(__itt_track_group_create(((group) ? UNICODE_AGNOSTIC(__itt_string_handle_create)(group) : nullptr), __itt_track_group_type_normal), UNICODE_AGNOSTIC(__itt_string_handle_create)(track), __itt_track_type_normal);\
    ITT_MAGIC_STATIC(itt_track_name);\
    itt_notify::ScopeTrack itt_track(itt_track_name);

//TODO: objects

} //namespace itt_notify

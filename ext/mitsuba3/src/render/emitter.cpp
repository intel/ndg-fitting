#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/endpoint.h>
#include <mitsuba/core/properties.h>

NAMESPACE_BEGIN(mitsuba)

MI_VARIANT Emitter<Float, Spectrum>::Emitter(const Properties &props)
    : Base(props) {
    m_num_parameters    = props.get<int>("num_parameters", 3);
    m_min_bounds        = props.get<ScalarVector3f>("min_bounds", 0.0f);
    m_range_bounds      = props.get<ScalarVector3f>("range_bounds", 1.0f);
}
MI_VARIANT Emitter<Float, Spectrum>::~Emitter() { }

MI_IMPLEMENT_CLASS_VARIANT(Emitter, Endpoint, "emitter")
MI_INSTANTIATE_CLASS(Emitter)
NAMESPACE_END(mitsuba)

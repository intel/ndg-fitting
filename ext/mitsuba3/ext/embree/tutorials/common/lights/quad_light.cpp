// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "light.h"

namespace embree {

struct QuadLight
{
  Light super;            //!< inherited light fields

  Vec3fa position;         //!< world-space corner position of the light
  Vec3fa edge1;            //!< vectors to adjacent corners
  Vec3fa edge2;            //!< vectors to adjacent corners
  Vec3fa radiance;         //!< RGB color and intensity of the QuadLight

  Vec3fa nnormal;          //!< negated normal, the direction that the QuadLight is not emitting; normalized
  float ppdf;             // probability to sample point on light = 1/area
};


// Implementation
//////////////////////////////////////////////////////////////////////////////

Light_SampleRes QuadLight_sample(const Light* super,
                                 const DifferentialGeometry& dg,
                                 const Vec2f& s)
{
  const QuadLight* self = (QuadLight*)super;
  Light_SampleRes res;

  // res position on light with density ppdf = 1/area
  // TODO: use solid angle sampling
  const Vec3fa pos = self->position + self->edge1 * s.x + self->edge2 * s.y;

  // extant light vector from the hit point
  const Vec3fa dir = pos - dg.P;
  const float dist = length(dir);

  // normalized light vector
  res.dir = dir / dist;
  res.dist = dist;

  // convert to pdf wrt. solid angle
  const float cosd = dot(self->nnormal, res.dir);
  res.pdf = self->ppdf * (dist * dist) / abs(cosd);

  // emit only to one side
  res.weight = cosd > 0.f ? self->radiance * rcp(res.pdf) : Vec3fa(0.f);

  return res;
}


// Exports (called from C++)
//////////////////////////////////////////////////////////////////////////////

//! Set the parameters of an ispc-side QuadLight object
extern "C" void QuadLight_set(void* super,
                          const Vec3fa& position,
                          const Vec3fa& edge2,
                          const Vec3fa& edge1,
                          const Vec3fa& radiance)
{
  QuadLight* self = (QuadLight*)super;
  self->position = position;
  self->edge1    = edge1;
  self->edge2    = edge2;
  self->radiance = radiance;

  const Vec3fa ndirection = cross(edge2, edge1);
  self->ppdf = rcp(length(ndirection));
  self->nnormal = ndirection * self->ppdf;
}

//! Create an ispc-side QuadLight object
extern "C" void* QuadLight_create()
{
  QuadLight* self = (QuadLight*) alignedMalloc(sizeof(QuadLight),16);

  Light_Constructor(&self->super);
  self->super.sample = QuadLight_sample;

  QuadLight_set(self,
                Vec3fa(0.f),
                Vec3fa(1.f, 0.f, 0.f),
                Vec3fa(0.f, 1.f, 0.f),
                Vec3fa(1.f));

  return self;
}

} // namespace embree

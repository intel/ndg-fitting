// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "distribution1d.h"
#include <algorithm>

namespace embree
{
  Distribution1D::Distribution1D()
    : size(0) {}

  Distribution1D::Distribution1D(const float* f, const size_t size_in) {
    init(f, size_in);
  }

  void Distribution1D::init(const float* f, const size_t size_in)
  {
    /*! create arrays */
    size = size_in;
    PDF.resize(size);
    CDF.resize(size+1);

    /*! accumulate the function f */
    CDF[0] = 0.0f;
    for (size_t i=1; i<size+1; i++)
      CDF[i] = CDF[i-1] + f[i-1];

    /*! compute reciprocal sum */
    float rcpSum = CDF[size] == 0.0f ? 0.0f : rcp(CDF[size]);

    /*! normalize the probability distribution and cumulative distribution */
    for (size_t i = 1; i<size+1; i++) {
      PDF[i-1] = f[i-1] * rcpSum * size;
      CDF[i] *= rcpSum;
    }
    CDF[size] = 1.0f;
  }

  float Distribution1D::sample(const float u) const
  {
    /*! coarse sampling of the distribution */
    const float* pointer = std::upper_bound(CDF.data(), CDF.data()+size, u);
    int index = clamp(int(pointer-CDF.data()-1),0,int(size)-1);

    /*! refine sampling linearly by assuming the distribution being a step function */
    const float dCDF = CDF[index+1] - CDF[index];
    if (dCDF == 0.0f) return float(index);
    float fraction = (u - CDF[index]) * rcp(dCDF);
    return float(index)+fraction;
  }
  
  float Distribution1D::pdf(const float p) const {
    return PDF[clamp(int(p*size),0,int(size)-1)];
  }
}


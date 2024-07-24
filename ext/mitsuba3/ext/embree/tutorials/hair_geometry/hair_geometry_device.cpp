// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../common/math/random_sampler.h"
#include "../common/tutorial/tutorial_device.h"
#include "../common/tutorial/scene_device.h"
#include "../common/tutorial/optics.h"

namespace embree {

/* accumulation buffer */
Vec3ff* g_accu = nullptr;
unsigned int g_accu_width = 0;
unsigned int g_accu_height = 0;
unsigned int g_accu_count = 0;
Vec3fa g_accu_vx;
Vec3fa g_accu_vy;
Vec3fa g_accu_vz;
Vec3fa g_accu_p;
extern "C" bool g_changed;
bool g_subdiv_mode = false;

/* light settings */
extern "C" Vec3fa g_dirlight_direction;
extern "C" Vec3fa g_dirlight_intensity;
extern "C" Vec3fa g_ambient_intensity;

/* hair material */
Vec3fa hair_K;
Vec3fa hair_dK;
Vec3fa hair_Kr;    //!< reflectivity of hair
Vec3fa hair_Kt;    //!< transparency of hair

void occlusionFilter(const RTCFilterFunctionNArguments* args);

/* scene data */
extern "C" ISPCScene* g_ispc_scene;
RTCScene g_scene = nullptr;

void convertTriangleMesh(ISPCTriangleMesh* mesh, RTCScene scene_out)
{
  RTCGeometry geom = rtcNewGeometry (g_device, RTC_GEOMETRY_TYPE_TRIANGLE);
  for (unsigned int t=0; t<mesh->numTimeSteps; t++) {
    rtcSetSharedGeometryBuffer(geom,RTC_BUFFER_TYPE_VERTEX,t,RTC_FORMAT_FLOAT3,mesh->positions[t],0,sizeof(Vertex),mesh->numVertices);
  }
  rtcSetSharedGeometryBuffer(geom,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT3,mesh->triangles,0,sizeof(ISPCTriangle),mesh->numTriangles);
  rtcSetGeometryOccludedFilterFunction(geom,occlusionFilter);
  rtcCommitGeometry(geom);
  rtcAttachGeometry(scene_out,geom);
  rtcReleaseGeometry(geom);
}

void convertHairSet(ISPCHairSet* hair, RTCScene scene_out)
{
  RTCGeometry geom = rtcNewGeometry (g_device, hair->type);
  for (unsigned int t=0; t<hair->numTimeSteps; t++) {
    rtcSetSharedGeometryBuffer(geom,RTC_BUFFER_TYPE_VERTEX,t,RTC_FORMAT_FLOAT4,hair->positions[t],0,sizeof(Vertex),hair->numVertices);
  }
  rtcSetSharedGeometryBuffer(geom,RTC_BUFFER_TYPE_INDEX,0,RTC_FORMAT_UINT,hair->hairs,0,sizeof(ISPCHair),hair->numHairs);
  if (hair->flags)
    rtcSetSharedGeometryBuffer(geom,RTC_BUFFER_TYPE_FLAGS,0,RTC_FORMAT_UCHAR,hair->flags,0,sizeof(char),hair->numHairs);
  rtcSetGeometryOccludedFilterFunction(geom,occlusionFilter);
  rtcSetGeometryTessellationRate(geom,(float)hair->tessellation_rate);
  rtcCommitGeometry(geom);
  rtcAttachGeometry(scene_out,geom);
  rtcReleaseGeometry(geom);
}

RTCScene convertScene(ISPCScene* scene_in)
{
  /* create scene */
  RTCScene scene_out = scene_in->scene;

  for (unsigned int i=0; i<scene_in->numGeometries; i++)
  {
    ISPCGeometry* geometry = scene_in->geometries[i];
    if (geometry->type == TRIANGLE_MESH)
      convertTriangleMesh((ISPCTriangleMesh*) geometry, scene_out);
    else if (geometry->type == CURVES)
      convertHairSet((ISPCHairSet*) geometry, scene_out);
  }

  /* commit changes to scene */
  rtcCommitScene (scene_out);

  return scene_out;
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

/*! Anisotropic power cosine microfacet distribution. */
struct AnisotropicBlinn {
  Vec3fa dx;       //!< x-direction of the distribution.
  float nx;        //!< Glossiness in x direction with range [0,infinity[ where 0 is a diffuse surface.
  Vec3fa dy;       //!< y-direction of the distribution.
  float ny;        //!< Exponent that determines the glossiness in y direction.
  Vec3fa dz;       //!< z-direction of the distribution.
  float norm1;     //!< Normalization constant for calculating the pdf for sampling.
  float norm2;     //!< Normalization constant for calculating the distribution.
  Vec3fa Kr,Kt;
  float side;
};

  /*! Anisotropic power cosine distribution constructor. */
inline void AnisotropicBlinn__Constructor(AnisotropicBlinn* This, const Vec3fa& Kr, const Vec3fa& Kt,
                                          const Vec3fa& dx, float nx, const Vec3fa& dy, float ny, const Vec3fa& dz)
{
  This->Kr = Kr;
  This->Kt = Kt;
  This->dx = dx;
  This->nx = nx;
  This->dy = dy;
  This->ny = ny;
  This->dz = dz;
  This->norm1 = sqrtf((nx+1)*(ny+1)) * float(one_over_two_pi);
  This->norm2 = sqrtf((nx+2)*(ny+2)) * float(one_over_two_pi);
  This->side = reduce_max(Kr)/(reduce_max(Kr)+reduce_max(Kt));
}

/*! Evaluates the power cosine distribution. \param wh is the half
 *  vector */
inline float AnisotropicBlinn__eval(const AnisotropicBlinn* This, const Vec3fa& wh)
{
  const float cosPhiH   = dot(wh, This->dx);
  const float sinPhiH   = dot(wh, This->dy);
  const float cosThetaH = dot(wh, This->dz);
  const float R = sqr(cosPhiH)+sqr(sinPhiH);
  if (R == 0.0f) return This->norm2;
  const float n = (This->nx*sqr(cosPhiH)+This->ny*sqr(sinPhiH))*rcp(R);
  return This->norm2 * pow(abs(cosThetaH), n);
}

/*! Samples the distribution. \param s is the sample location
 *  provided by the caller. */
inline Vec3ff AnisotropicBlinn__sample(const AnisotropicBlinn* This, const float sx, const float sy)
{
  const float phi =float(two_pi)*sx;
  const float sinPhi0 = sqrtf(This->nx+1)*sinf(phi);
  const float cosPhi0 = sqrtf(This->ny+1)*cosf(phi);
  const float norm = rsqrt(sqr(sinPhi0)+sqr(cosPhi0));
  const float sinPhi = sinPhi0*norm;
  const float cosPhi = cosPhi0*norm;
  const float n = This->nx*sqr(cosPhi)+This->ny*sqr(sinPhi);
  const float cosTheta = powf(sy,rcp(n+1));
  const float sinTheta = cos2sin(cosTheta);
  const float pdf = This->norm1*powf(cosTheta,n);
  const Vec3fa h = Vec3fa(cosPhi * sinTheta, sinPhi * sinTheta, cosTheta);
  const Vec3fa wh = h.x*This->dx + h.y*This->dy + h.z*This->dz;
  return Vec3ff(wh,pdf);
}

inline Vec3fa AnisotropicBlinn__eval(const AnisotropicBlinn* This, const Vec3fa& wo, const Vec3fa& wi)
{
  const float cosThetaI = dot(wi,This->dz);

  /* reflection */
  if (cosThetaI > 0.0f) {
    const Vec3fa wh = normalize(wi + wo);
    return This->Kr * AnisotropicBlinn__eval(This,wh) * abs(cosThetaI);
  }

  /* transmission */
  else {
    const Vec3fa wh = normalize(reflect(wi,This->dz) + wo);
    return This->Kt * AnisotropicBlinn__eval(This,wh) * abs(cosThetaI);
  }
}

inline Vec3fa AnisotropicBlinn__sample(const AnisotropicBlinn* This, const Vec3fa& wo, Vec3ff& wi, const float sx, const float sy, const float sz)
{
  //wi = Vec3fa(reflect(normalize(wo),normalize(dz)),1.0f); return Kr;
  //wi = Vec3fa(neg(wo),1.0f); return Kt;
  const Vec3ff wh = AnisotropicBlinn__sample(This,sx,sy);
  //if (dot(wo,wh) < 0.0f) return Vec3fa(0.0f,0.0f);

  /* reflection */
  if (sz < This->side) {
    wi = Vec3ff(reflect(wo,Vec3fa(wh)),wh.w*This->side);
    const float cosThetaI = dot(Vec3fa(wi),This->dz);
    return This->Kr * AnisotropicBlinn__eval(This,Vec3fa(wh)) * abs(cosThetaI);
  }

  /* transmission */
  else {
    wi = Vec3ff(reflect(reflect(wo,Vec3fa(wh)),This->dz),wh.w*(1-This->side));
    const float cosThetaI = dot(Vec3fa(wi),This->dz);
    return This->Kt * AnisotropicBlinn__eval(This,Vec3fa(wh)) * abs(cosThetaI);
  }
}

typedef Vec3fa* uniform_Vec3fa_ptr;

inline Vec3fa evalBezier(const unsigned int geomID, const unsigned int primID, const float t)
{
  const float t0 = 1.0f - t, t1 = t;
  const ISPCHairSet* hair = (const ISPCHairSet*) g_ispc_scene->geometries[geomID];
  const Vec3fa* vertices = hair->positions[0];
  const ISPCHair* hairs = hair->hairs;

  const int i = hairs[primID].vertex;
  const Vec3fa p00 = vertices[i+0];
  const Vec3fa p01 = vertices[i+1];
  const Vec3fa p02 = vertices[i+2];
  const Vec3fa p03 = vertices[i+3];

  const Vec3fa p10 = p00 * t0 + p01 * t1;
  const Vec3fa p11 = p01 * t0 + p02 * t1;
  const Vec3fa p12 = p02 * t0 + p03 * t1;
  const Vec3fa p20 = p10 * t0 + p11 * t1;
  const Vec3fa p21 = p11 * t0 + p12 * t1;
  const Vec3fa p30 = p20 * t0 + p21 * t1;

  return p30;
  //tangent = p21-p20;
}

/* occlusion filter function */
void occlusionFilter(const RTCFilterFunctionNArguments* args)

{
  IntersectContext* context = (IntersectContext*) args->context;
  Vec3fa* transparency = (Vec3fa*) context->userRayExt;
  if (!transparency) return;
    
  int* valid_i = args->valid;
  struct RTCHitN* hit = args->hit;
  const unsigned int N = args->N;
  assert(N == 1);
  bool valid = *((int*) valid_i);
  if (!valid) return;
 
  /* make all surfaces opaque */
  unsigned int geomID = RTCHitN_geomID(hit,N,0);
  ISPCGeometry* geometry = g_ispc_scene->geometries[geomID];
  if (geometry->type == TRIANGLE_MESH) {
    *transparency = Vec3fa(0.0f);
    return;
  }
  Vec3fa T = hair_Kt;
  T = T * *transparency;
  *transparency = T;
  if (eq(T,Vec3fa(0.0f)))
    ;
  else
    valid_i[0] = 0;
}

Vec3fa occluded(RTCScene scene, IntersectContext* context, Ray& ray)
{
  Vec3fa transparency = Vec3fa(1.0f);
  context->userRayExt = &transparency;
  
  ray.geomID = RTC_INVALID_GEOMETRY_ID;
  ray.primID = RTC_INVALID_GEOMETRY_ID;
  ray.mask = -1;
  rtcOccluded1(scene,&context->context,RTCRay_(ray));

  return transparency;
}

/* task that renders a single screen tile */
Vec3fa renderPixelStandard(float x, float y, const ISPCCamera& camera, RayStats& stats)
{
  RandomSampler sampler;
  RandomSampler_init(sampler, (int)x, (int)y, g_accu_count);
  x += RandomSampler_get1D(sampler);
  y += RandomSampler_get1D(sampler);
  float time = RandomSampler_get1D(sampler);

  IntersectContext context;
  InitIntersectionContext(&context);
  
  /* initialize ray */
  Ray ray(Vec3fa(camera.xfm.p), Vec3fa(normalize(x*camera.xfm.l.vx + y*camera.xfm.l.vy + camera.xfm.l.vz)), 0.0f, inf);

  Vec3fa color = Vec3fa(0.0f);
  Vec3fa weight = Vec3fa(1.0f);
  unsigned int depth = 0;

  while (true)
  {
    /* terminate ray path */
    if (reduce_max(weight) < 0.01 || depth > 20)
      return color;

    /* intersect ray with scene and gather all hits */
    rtcIntersect1(g_scene,&context.context,RTCRayHit_(ray));
    RayStats_addRay(stats);

    /* exit if we hit environment */
    if (ray.geomID == RTC_INVALID_GEOMETRY_ID)
      return color + weight*Vec3fa(g_ambient_intensity);

    /* calculate transmissivity of hair */
    AnisotropicBlinn brdf;
    float eps = 0.0001f;

    ISPCGeometry* geometry = g_ispc_scene->geometries[ray.geomID];
    if (geometry->type == CURVES)
    {
      /* calculate tangent space */
      const Vec3fa dx = normalize(ray.Ng);
      const Vec3fa dy = normalize(cross(ray.dir,dx));
      const Vec3fa dz = normalize(cross(dy,dx));

      /* generate anisotropic BRDF */
      AnisotropicBlinn__Constructor(&brdf,hair_Kr,hair_Kt,dx,20.0f,dy,2.0f,dz);
      brdf.Kr = hair_Kr;
    }
    else if (geometry->type == TRIANGLE_MESH)
    {
      if (dot(ray.dir,ray.Ng) > 0) ray.Ng = neg(ray.Ng);

      /* calculate tangent space */
      const Vec3fa dz = normalize(ray.Ng);
      const Vec3fa dx = normalize(cross(dz,ray.dir));
      const Vec3fa dy = normalize(cross(dz,dx));

      /* generate isotropic BRDF */
      AnisotropicBlinn__Constructor(&brdf,Vec3fa(1.0f),Vec3fa(0.0f),dx,1.0f,dy,1.0f,dz);
    }
    else
      return color;

    /* sample directional light */
    Ray shadow(ray.org + ray.tfar*ray.dir, neg(Vec3fa(g_dirlight_direction)), eps, inf, time);
    Vec3fa T = occluded(g_scene,&context,shadow);
    RayStats_addShadowRay(stats);
    Vec3fa c = AnisotropicBlinn__eval(&brdf,neg(ray.dir),neg(Vec3fa(g_dirlight_direction)));
    color = color + weight*c*T*Vec3fa(g_dirlight_intensity);

#if 1
    /* sample BRDF */
    Vec3ff wi;
    float ru = RandomSampler_get1D(sampler);
    float rv = RandomSampler_get1D(sampler);
    float rw = RandomSampler_get1D(sampler);
    c = AnisotropicBlinn__sample(&brdf,neg(ray.dir),wi,ru,rv,rw);
    if (wi.w <= 0.0f) return color;

    /* calculate secondary ray and offset it out of the hair */
    float sign = dot(Vec3fa(wi),brdf.dz) < 0.0f ? -1.0f : 1.0f;
    ray.org = Vec3ff(ray.org + ray.tfar*ray.dir + sign*eps*brdf.dz);
    ray.dir = Vec3ff(wi);
    ray.tnear() = 0.001f;
    ray.tfar = inf;
    ray.geomID = RTC_INVALID_GEOMETRY_ID;
    ray.primID = RTC_INVALID_GEOMETRY_ID;
    ray.mask = -1;
    ray.time() = time;
    weight = weight * c/wi.w;

#else

    /* continue with transparency ray */
    ray.geomID = RTC_INVALID_GEOMETRY_ID;
    ray.tnear() = 1.001f*ray.tfar;
    ray.tfar = inf;
    weight *= brdf.Kt;

#endif

    depth++;
  }
  return color;
}

/* renders a single screen tile */
void renderTileStandard(int taskIndex,
                        int threadIndex,
                        int* pixels,
                        const unsigned int width,
                        const unsigned int height,
                        const float time,
                        const ISPCCamera& camera,
                        const int numTilesX,
                        const int numTilesY)
{
  const unsigned int tileY = taskIndex / numTilesX;
  const unsigned int tileX = taskIndex - tileY * numTilesX;
  const unsigned int x0 = tileX * TILE_SIZE_X;
  const unsigned int x1 = min(x0+TILE_SIZE_X,width);
  const unsigned int y0 = tileY * TILE_SIZE_Y;
  const unsigned int y1 = min(y0+TILE_SIZE_Y,height);

  for (unsigned int y=y0; y<y1; y++) for (unsigned int x=x0; x<x1; x++)
  {
    /* calculate pixel color */
    Vec3fa color = renderPixelStandard((float)x,(float)y,camera,g_stats[threadIndex]);

    /* write color to framebuffer */
    Vec3ff accu_color = g_accu[y*width+x] + Vec3ff(color.x,color.y,color.z,1.0f); g_accu[y*width+x] = accu_color;
    float f = rcp(max(0.001f,accu_color.w));
    unsigned int r = (unsigned int) (255.01f * clamp(accu_color.x*f,0.0f,1.0f));
    unsigned int g = (unsigned int) (255.01f * clamp(accu_color.y*f,0.0f,1.0f));
    unsigned int b = (unsigned int) (255.01f * clamp(accu_color.z*f,0.0f,1.0f));
    pixels[y*width+x] = (b << 16) + (g << 8) + r;
  }
}

/* task that renders a single screen tile */
void renderTileTask (int taskIndex, int threadIndex, int* pixels,
                         const unsigned int width,
                         const unsigned int height,
                         const float time,
                         const ISPCCamera& camera,
                         const int numTilesX,
                         const int numTilesY)
{
  renderTileStandard(taskIndex,threadIndex,pixels,width,height,time,camera,numTilesX,numTilesY);
}

/* called by the C++ code for initialization */
extern "C" void device_init (char* cfg)
{
  /* initialize last seen camera */
  g_accu_vx = Vec3fa(0.0f);
  g_accu_vy = Vec3fa(0.0f);
  g_accu_vz = Vec3fa(0.0f);
  g_accu_p  = Vec3fa(0.0f);

  /* initialize hair colors */
  hair_K  = Vec3fa(0.8f,0.57f,0.32f);
  hair_dK = Vec3fa(0.1f,0.12f,0.08f);
  hair_Kr = 0.2f*hair_K;    //!< reflectivity of hair
  hair_Kt = 0.8f*hair_K;    //!< transparency of hair

  /* create scene */
  g_scene = convertScene(g_ispc_scene);
}

extern "C" void renderFrameStandard (int* pixels,
                          const unsigned int width,
                          const unsigned int height,
                          const float time,
                          const ISPCCamera& camera)
{
  /* render frame */
  const int numTilesX = (width +TILE_SIZE_X-1)/TILE_SIZE_X;
  const int numTilesY = (height+TILE_SIZE_Y-1)/TILE_SIZE_Y;
  parallel_for(size_t(0),size_t(numTilesX*numTilesY),[&](const range<size_t>& range) {
    const int threadIndex = (int)TaskScheduler::threadIndex();
    for (size_t i=range.begin(); i<range.end(); i++)
      renderTileTask((int)i,threadIndex,pixels,width,height,time,camera,numTilesX,numTilesY);
  }); 
}

/* called by the C++ code to render */
extern "C" void device_render (int* pixels,
                           const unsigned int width,
                           const unsigned int height,
                           const float time,
                           const ISPCCamera& camera)
{
  /* create accumulator */
  if (g_accu_width != width || g_accu_height != height) {
    g_accu = (Vec3ff*) alignedMalloc(width*height*sizeof(Vec3ff),16);
    g_accu_width = width;
    g_accu_height = height;
    for (unsigned int i=0; i<width*height; i++)
      g_accu[i] = Vec3ff(0.0f);
  }

  /* reset accumulator */
  bool camera_changed = g_changed; g_changed = false;
  camera_changed |= ne(g_accu_vx,camera.xfm.l.vx); g_accu_vx = camera.xfm.l.vx;
  camera_changed |= ne(g_accu_vy,camera.xfm.l.vy); g_accu_vy = camera.xfm.l.vy;
  camera_changed |= ne(g_accu_vz,camera.xfm.l.vz); g_accu_vz = camera.xfm.l.vz;
  camera_changed |= ne(g_accu_p, camera.xfm.p   ); g_accu_p  = camera.xfm.p;
  g_accu_count++;
  if (camera_changed) {
    g_accu_count=0;
    for (unsigned int i=0; i<width*height; i++)
      g_accu[i] = Vec3ff(0.0f);
  }
}

/* called by the C++ code for cleanup */
extern "C" void device_cleanup ()
{
  rtcReleaseScene (g_scene); g_scene = nullptr;
  alignedFree(g_accu); g_accu = nullptr;
  g_accu_width = 0;
  g_accu_height = 0;
  g_accu_count = 0;
}

} // namespace embree

// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZETOOLS_H
#define RASTERIZETOOLS_H

#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "cudaMat4.h"

struct triangle {
  glm::vec4 p0;
  glm::vec4 p1;
  glm::vec4 p2;
  glm::vec3 c0;
  glm::vec3 c1;
  glm::vec3 c2;
};

struct fragment{
  glm::vec3 color;
  glm::vec3 normal;
  glm::vec3 position;
};

//Multiplies a cudaMat4 matrix and a vec4
__host__ __device__ glm::vec3 multiplyMV(cudaMat4 m, glm::vec4 v){
  glm::vec3 r(1,1,1);
  r.x = (m.x.x*v.x)+(m.x.y*v.y)+(m.x.z*v.z)+(m.x.w*v.w);
  r.y = (m.y.x*v.x)+(m.y.y*v.y)+(m.y.z*v.z)+(m.y.w*v.w);
  r.z = (m.z.x*v.x)+(m.z.y*v.y)+(m.z.z*v.z)+(m.z.w*v.w);
  return r;
}

//LOOK: finds the axis aligned bounding box for a given triangle
__host__ __device__ void getAABBForTriangle(triangle tri, glm::vec3& minpoint, glm::vec3& maxpoint){
  minpoint = glm::vec3(min(min(tri.p0.x, tri.p1.x),tri.p2.x), 
        min(min(tri.p0.y, tri.p1.y),tri.p2.y),
        min(min(tri.p0.z, tri.p1.z),tri.p2.z));
  maxpoint = glm::vec3(max(max(tri.p0.x, tri.p1.x),tri.p2.x), 
        max(max(tri.p0.y, tri.p1.y),tri.p2.y),
        max(max(tri.p0.z, tri.p1.z),tri.p2.z));
}

//LOOK: calculates the signed area of a given triangle
__host__ __device__ float calculateSignedArea(triangle tri){
  return 0.5*((tri.p2.x - tri.p0.x)*(tri.p1.y - tri.p0.y) - (tri.p1.x - tri.p0.x)*(tri.p2.y - tri.p0.y));
}

//LOOK: helper function for calculating barycentric coordinates
__host__ __device__ float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, triangle tri){
  triangle baryTri;
  baryTri.p0 = glm::vec4 (glm::vec3(a,0), 1.0); baryTri.p1 = glm::vec4 (glm::vec3(b,0), 1.0); baryTri.p2 = glm::vec4 (glm::vec3(c,0), 1.0);
  return calculateSignedArea(baryTri)/calculateSignedArea(tri);
}

//LOOK: calculates barycentric coordinates
__host__ __device__ glm::vec3 calculateBarycentricCoordinate(triangle tri, glm::vec2 point){
  float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri.p0.x,tri.p0.y), point, glm::vec2(tri.p2.x,tri.p2.y), tri);
  float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri.p0.x,tri.p0.y), glm::vec2(tri.p1.x,tri.p1.y), point, tri);
  float alpha = 1.0-beta-gamma;
  return glm::vec3(alpha,beta,gamma);
}

//LOOK: checks if a barycentric coordinate is within the boundaries of a triangle
__host__ __device__ bool isBarycentricCoordInBounds(glm::vec3 barycentricCoord){
   return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
          barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
          barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

//LOOK: for a given barycentric coordinate, return the corresponding z position on the triangle
__host__ __device__ float getZAtCoordinate(glm::vec3 barycentricCoord, triangle tri){
  return -(barycentricCoord.x*tri.p0.z + barycentricCoord.y*tri.p1.z + barycentricCoord.z*tri.p2.z);
}

//__host__ __device__ void clipToScreenTransform (triangle &primitive, glm::vec2 resolution)
//{
//	// Convert clip space coordinates to NDC (a.k.a. Perspective divide).
//	primitive.p0.x /= primitive.p0.w;
//	primitive.p0.y /= primitive.p0.w;
//	primitive.p0.z /= primitive.p0.w;
//
//	primitive.p1.x /= primitive.p1.w;
//	primitive.p1.y /= primitive.p1.w;
//	primitive.p1.z /= primitive.p1.w;
//
//	primitive.p2.x /= primitive.p2.w;
//	primitive.p2.y /= primitive.p2.w;
//	primitive.p2.z /= primitive.p2.w;
//
//	// Rescale NDC to be in the range 0.0 to 1.0.
//	primitive.p0.x += 1.0f;
//	primitive.p0.x /= 2.0f;
//	primitive.p0.y += 1.0f;
//	primitive.p0.y /= 2.0f;
//	primitive.p0.z += 1.0f;
//	primitive.p0.z /= 2.0f;
//
//	primitive.p1.x += 1.0f;
//	primitive.p1.x /= 2.0f;
//	primitive.p1.y += 1.0f;
//	primitive.p1.y /= 2.0f;
//	primitive.p1.z += 1.0f;
//	primitive.p1.z /= 2.0f;
//
//	primitive.p2.x += 1.0f;
//	primitive.p2.x /= 2.0f;
//	primitive.p2.y += 1.0f;
//	primitive.p2.y /= 2.0f;
//	primitive.p2.z += 1.0f;
//	primitive.p2.z /= 2.0f;
//
//	// Now multiply with resolution to get screen co-ordinates.
//	primitive.p0.x *= resolution.x;
//	primitive.p0.y *= resolution.y;
//	  
//	primitive.p1.x *= resolution.x;
//	primitive.p1.y *= resolution.y;
//	  
//	primitive.p2.x *= resolution.x;
//	primitive.p2.y *= resolution.y;
//}
//
//__host__ __device__ void screenToClipTransform (triangle &primitive, glm::vec2 resolution)
//{
//	// Divide with resolution to get NDC.
//	primitive.p0.x /= resolution.x;
//	primitive.p0.y /= resolution.y;
//	  
//	primitive.p1.x /= resolution.x;
//	primitive.p1.y /= resolution.y;
//	  
//	primitive.p2.x /= resolution.x;
//	primitive.p2.y /= resolution.y;
//
//	// Rescale NDC to be in the range -1.0 to 1.0.
//	primitive.p0.x *= 2.0f;
//	primitive.p0.x -= 1.0f;
//	primitive.p0.y *= 2.0f;
//	primitive.p0.y -= 1.0f;
//	primitive.p0.z *= 2.0f;
//	primitive.p0.z -= 1.0f;
//
//	primitive.p1.x *= 2.0f;
//	primitive.p1.x -= 1.0f;
//	primitive.p1.y *= 2.0f;
//	primitive.p1.y -= 1.0f;
//	primitive.p1.z *= 2.0f;
//	primitive.p1.z -= 1.0f;
//
//	primitive.p2.x *= 2.0f;
//	primitive.p2.x -= 1.0f;
//	primitive.p2.y *= 2.0f;
//	primitive.p2.y -= 1.0f;
//	primitive.p2.z *= 2.0f;
//	primitive.p2.z -= 1.0f;
//
//	// Convert NDC to clip space coordinates
//	primitive.p0.x *= primitive.p0.w;
//	primitive.p0.y *= primitive.p0.w;
//	primitive.p0.z *= primitive.p0.w;
//
//	primitive.p1.x *= primitive.p1.w;
//	primitive.p1.y *= primitive.p1.w;
//	primitive.p1.z *= primitive.p1.w;
//
//	primitive.p2.x *= primitive.p2.w;
//	primitive.p2.y *= primitive.p2.w;
//	primitive.p2.z *= primitive.p2.w;
//}

#endif
// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZETOOLS_H
#define RASTERIZETOOLS_H

#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "cudaMat4.h"
#include "rasterizeStructs.h"

#define MIN(a,b) (a<b?a:b);

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
  minpoint = glm::vec3(glm::min(glm::min(tri.v0.pos.x, tri.v1.pos.x),tri.v2.pos.x), 
        glm::min(glm::min(tri.v0.pos.y, tri.v1.pos.y),tri.v2.pos.y),
        glm::min(glm::min(tri.v0.pos.z, tri.v1.pos.z),tri.v2.pos.z));
  maxpoint = glm::vec3(glm::max(glm::max(tri.v0.pos.x, tri.v1.pos.x),tri.v2.pos.x), 
        glm::max(glm::max(tri.v0.pos.y, tri.v1.pos.y),tri.v2.pos.y),
        glm::max(glm::max(tri.v0.pos.z, tri.v1.pos.z),tri.v2.pos.z));
}


//LOOK: finds the axis aligned bounding box for a given triangle
__host__ __device__ void getCompact2DAABBForTriangle(triangle tri, glm::vec4& minXminYmaxXmaxY){
  minXminYmaxXmaxY = glm::vec4(glm::min(glm::min(tri.v0.pos.x, tri.v1.pos.x),tri.v2.pos.x), 
        glm::min(glm::min(tri.v0.pos.y, tri.v1.pos.y),tri.v2.pos.y),
        glm::max(glm::max(tri.v0.pos.x, tri.v1.pos.x),tri.v2.pos.x), 
        glm::max(glm::max(tri.v0.pos.y, tri.v1.pos.y),tri.v2.pos.y));
}


//LOOK: calculates the signed area of a given triangle
__host__ __device__ float calculateSignedArea(triangle tri){
  return 0.5*((tri.v2.pos.x - tri.v0.pos.x)*(tri.v1.pos.y - tri.v0.pos.y) - (tri.v1.pos.x - tri.v0.pos.x)*(tri.v2.pos.y - tri.v0.pos.y));
}

//LOOK: helper function for calculating barycentric coordinates
__host__ __device__ float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, triangle tri){
  triangle baryTri;
  baryTri.v0.pos = glm::vec3(a,0); baryTri.v1.pos = glm::vec3(b,0); baryTri.v2.pos = glm::vec3(c,0);
  return calculateSignedArea(baryTri)/calculateSignedArea(tri);
}

//LOOK: calculates barycentric coordinates
__host__ __device__ glm::vec3 calculateBarycentricCoordinate(triangle tri, glm::vec2 point){
  float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri.v0.pos.x,tri.v0.pos.y), point, glm::vec2(tri.v2.pos.x,tri.v2.pos.y), tri);
  float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri.v0.pos.x,tri.v0.pos.y), glm::vec2(tri.v1.pos.x,tri.v1.pos.y), point, tri);
  float alpha = 1.0-beta-gamma;
  return glm::vec3(alpha,beta,gamma);
}

//LOOK: checks if a barycentric coordinate is within the boundaries of a triangle
__host__ __device__ bool isBarycentricCoordInBounds(glm::vec3 barycentricCoord){
   return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
          barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
          barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}


//Converts a triangle from clip space to a screen resolution mapped space 
//From (-1:1,-1:1,-1:1) to (0:w, 0:h, 0:1)
__host__ __device__ void transformTriToScreenSpace(triangle &tri, glm::vec2 resolution)
{
	//Scale and shift x
	tri.v0.pos.x = (tri.v0.pos.x+1.0)*0.5f*resolution.x;
	tri.v1.pos.x = (tri.v1.pos.x+1.0)*0.5f*resolution.x;
	tri.v2.pos.x = (tri.v2.pos.x+1.0)*0.5f*resolution.x;

	//Scale and shift y
	tri.v0.pos.y = (tri.v0.pos.y+1.0)*0.5f*resolution.y;
	tri.v1.pos.y = (tri.v1.pos.y+1.0)*0.5f*resolution.y;
	tri.v2.pos.y = (tri.v2.pos.y+1.0)*0.5f*resolution.y;

	//Scale and shift z
	tri.v0.pos.z = (tri.v0.pos.z+1.0)*0.5f;
	tri.v1.pos.z = (tri.v1.pos.z+1.0)*0.5f;
	tri.v2.pos.z = (tri.v2.pos.z+1.0)*0.5f;
}

//Returns true if the AABB defined by these two points overlaps with clip region (-1:1, -1:1, -1:1)
__host__ __device__ bool isAABBInClipSpace(glm::vec3 minpoint, glm::vec3 maxpoint)
{
	 if (minpoint.x > 1.0 || -1.0 > maxpoint.x)
		 return false;
	 if (minpoint.y > 1.0 || -1.0 > maxpoint.y)
		 return false;
	 if (minpoint.z > 1.0 || -1.0 > maxpoint.z)
		 return false;
	 

	 return true;
}


//Checks for 2D AABB intersection in memory efficient
__host__ __device__ bool doCompactAABBsintersect(glm::vec4 aabb1, glm::vec4 aabb2)
{
	//If 1.minX > 2.maxX || 2.minX > 1.maxX
	if (aabb1.x > aabb2.z || aabb2.x > aabb1.z)
		 return false;
	
	//If 1.minY > 2.maxY || 2.minY > 1.maxY
	if (aabb1.y > aabb2.w || aabb2.y > aabb1.w)
		 return false;

	 return true;
}
__host__ __device__ bool isAABBInBin(glm::vec3 minpoint, glm::vec3 maxpoint, int binXMin, int binXMax, int binYMin, int binYMax)
{
	if (minpoint.x > binXMax  || binXMin > maxpoint.x)
		 return false;
	 if (minpoint.y > binYMax || binYMin> maxpoint.y)
		 return false;

	 return true;
}

#endif
// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZETOOLS_H
#define RASTERIZETOOLS_H

#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"
#include "cudaMat4.h"
#define TRI_TEST_EPSILON 0.0001
#define KBUFFER_SIZE 8

struct triangle {
 
	//vertices in SCREEN SPACE
glm::vec3 p0;
  glm::vec3 p1;
  glm::vec3 p2;
  glm::vec3 c0;
  glm::vec3 c1;
  glm::vec3 c2;
  glm::vec3 n0;

  //vertices in MODEL SPACE
  glm::vec3 modelspace_p0;
  glm::vec3 modelspace_p1;
  glm::vec3 modelspace_p2;

  //normals in MODEL SPACE
  glm::vec3 modelspace_n0;
  glm::vec3 modelspace_n1;
  glm::vec3 modelspace_n2;
};

struct fragment{
  glm::vec3 color;
  glm::vec3 normal;
  glm::vec3 modelNormal;
  glm::vec3 position;
  glm::vec3 modelPosition; //position in modelspace
  int triIdx;
};

struct fragmentArray{
	fragment fragArray[KBUFFER_SIZE];
	int depth[KBUFFER_SIZE];
	int numFragments;
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
  baryTri.p0 = glm::vec3(a,0); baryTri.p1 = glm::vec3(b,0); baryTri.p2 = glm::vec3(c,0);
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
   return barycentricCoord.x >= -TRI_TEST_EPSILON && barycentricCoord.x <= 1.0 + TRI_TEST_EPSILON &&
          barycentricCoord.y >= -TRI_TEST_EPSILON && barycentricCoord.y <= 1.0 + TRI_TEST_EPSILON &&
          barycentricCoord.z >= -TRI_TEST_EPSILON && barycentricCoord.z <= 1.0 + TRI_TEST_EPSILON ;
}

//LOOK: for a given barycentric coordinate, return the corresponding z position on the triangle
__host__ __device__ float getZAtCoordinate(glm::vec3 barycentricCoord, triangle tri){
  return -(barycentricCoord.x*tri.p0.z + barycentricCoord.y*tri.p1.z + barycentricCoord.z*tri.p2.z);
}

//for 3 points p0, p1, and p2 corresponding to p0, p1, and p2 in the triangle, interpolate based on barycentric coords.
__host__ __device__ glm::vec3 interpVec3(glm::vec3 barycentricCoord, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2){
	return (barycentricCoord.x*p0 + barycentricCoord.y*p1 + barycentricCoord.z*p2);
}

#endif
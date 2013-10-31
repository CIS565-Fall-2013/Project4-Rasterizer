// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef STRUCTS_H
#define STRUCTS_H

#include "glm/glm.hpp"
#include "cudaMat4.h"
#include <cuda_runtime.h>
#include <string>

struct triangle {
  glm::vec3 p0;
  glm::vec3 p1;
  glm::vec3 p2;
  glm::vec3 c0;
  glm::vec3 c1;
  glm::vec3 c2;
  glm::vec3 n0;
  glm::vec3 n1;
  glm::vec3 n2;
};

struct fragment {
  glm::vec3 color;
  glm::vec3 normal;
  glm::vec3 position;
};

struct camera {
  glm::vec3 position;
  glm::vec3 view;
  glm::vec3 up;
  glm::vec3 right;
  float fovy;
};

struct ray {
  glm::vec3 position;
  glm::vec3 color;
};

#endif // end of CUDASTRUCTS_H
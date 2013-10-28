#ifndef RASTERIZESTRUCTS_H
#define RASTERIZESTRUCTS_H

#include "glm/glm.hpp"

struct triangle {
	glm::vec3 p0;
	glm::vec3 p1;
	glm::vec3 p2;
	glm::vec3 c0;
	glm::vec3 c1;
	glm::vec3 c2;
};

struct fragment{
	glm::vec3 color;
	glm::vec3 normal;
	glm::vec3 position;
};

struct camera {
	float fovy;
	glm::vec2 resolution;
	glm::vec3 position;
	glm::vec3 viewDir;
	glm::vec3 up;
	glm::mat4 view;
	glm::mat4 projection;
	glm::mat4 viewport;
};

#endif
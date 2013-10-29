#pragma once
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

struct uniforms{
	glm::mat4 viewTransform;
	glm::mat4 perspectiveTransform;
};

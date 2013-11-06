#ifndef RASTERIZESTRUCTS_H
#define RASTERIZESTRUCTS_H

#include "glm/glm.hpp"

struct triangle {
	glm::vec3 p0;   // point in screen space
	glm::vec3 p1; 
	glm::vec3 p2;
	glm::vec3 c0;   // color
	glm::vec3 c1;
	glm::vec3 c2;
	glm::vec3 n0;   // normal in world (camera) space
	glm::vec3 n1;
	glm::vec3 n2;
	glm::vec3 pw0;  // point in world (camera) space
	glm::vec3 pw1;
	glm::vec3 pw2;
	bool toDiscard; // true for back faces, false for front faces
};

struct fragment{
	glm::vec3 color;
	glm::vec3 normal;
	glm::vec3 position;
	float depth;
	int locked;
};

struct camera {
	float fovy;
	float zNear;
	float zFar;
	glm::vec2 resolution;
	glm::vec3 position;
	glm::vec3 up;
	glm::mat4 view;
	glm::mat4 projection;
};

struct light {
	glm::vec3 position;
	glm::vec3 color;
};

#endif
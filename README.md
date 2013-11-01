-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------
Due Thursday 10/31/2012
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
For this project, I implemented a GPU rasterizer that reads and rasterizes an .obj
file. The pipeline is as follows:

1. Transform the .obj vertices from model space to screen space using a series of 
matrix transformations. 

2. Do primitive assembly, i.e. take three vertices at a time and group them into 
triangles. Calculate normals (I keep them in camera space).

3. Rasterize per primitive -- for each primitive, find out which fragments it contains. 

4. Fragment shading -- Do diffuse shading for fragments that are contained by a primitive.

I implemented the following extra features:

* Color interpolation between points on a primitive

* Back-face culling -- I throw away all triangles that don't face the eye

* Mouse based interactive camera support

-------------------------------------------------------------------------------
Images:
-------------------------------------------------------------------------------



[Here](https://vimeo.com/78320271) is a video of the cow and the Stanford dragon
being rasterized. 

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
Rasterizing per primitive VS per fragment

![alt text](./renders/cow_normal_diffuse.png "normal and diffuse")

![alt text](./renders/ "diffuse")

![alt text](./renders/dragon_specular.png "specular")

Using backface culling



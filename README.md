-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------


INTRODUCTION:
-------------------------------------------------------------------------------
In this project, a simplified CUDA based implementation of a standard rasterized graphics pipeline, similar to the OpenGL pipeline has been implemented. In this project I have implemented vertex shading, primitive assembly, perspective transformation, rasterization, fragment shading, and the resulting fragments are written to a framebuffer.

The following stages of the graphics pipeline and features have been implemented:

* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs/NBOs
* Perspective Transformation
* Rasterization through barycentric co-ordinates and Atomic exchange
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple Phong lighting/shading scheme.

3 of the following additional features have been implemented:

* Interactive mouse
  Here the interactive camera from my path tracer has been extended to the rasterizer and the model,view,projection matrix is 
  recalculated each time a mouse movement is detected

* Backface culling
  While implementing backface culling, a dot product between the eye vector and the normal of the primitive is taken and the value obtained is
  checked if it is positive or negative. If negative the primitive is removed from the array and using stream compaction and dead primitives are
  removed from the list and hence I get a significant speed up during the rasterization process.

* Correct color interpretation between points on a primitive
  While implementing proper color interpretation, the smooth normals of the OBJ are obtained and then a new NBO is created and passed onto the cudarasterizecore() 
  similar to the VBO and then these normals are multiplied by the model matrix and stored in the primitive. Later these are interpolated using the barycentric co-ordinates to get the proper color at that point. 


-------------------------------------------------------------------------------
VIDEO:
-------------------------------------------------------------------------------
[Rasterizer Video](http://www.youtube.com/watch?v=5xLlpkohLzw&feature=youtu.be)
-------------------------------------------------------------------------------
RENDERS:
-------------------------------------------------------------------------------
Depth Map of OBJ
![alt tag](https://raw.github.com/vivreddy/Project4-Rasterizer/master/renders/depths.png)

Normals of OBJ
![alt tag](https://raw.github.com/vivreddy/Project4-Rasterizer/master/renders/normals.png)

Cow model with Back face culling
![alt tag](https://raw.github.com/vivreddy/Project4-Rasterizer/master/renders/phongcow.png)

Stanford bunny model with Back face culling
![alt tag](https://raw.github.com/vivreddy/Project4-Rasterizer/master/renders/phongbunny.png)

Dragon model with Back face culling
![alt tag](https://raw.github.com/vivreddy/Project4-Rasterizer/master/renders/phongdragon.png)

Buddha model with Back face culling
![alt tag](https://raw.github.com/vivreddy/Project4-Rasterizer/master/renders/phongbuddha.png)



-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------

![alt tag](https://raw.github.com/vivreddy/Project4-Rasterizer/master/renders/table.png)

![alt tag](https://raw.github.com/vivreddy/Project4-Rasterizer/master/renders/graph.png)

For the performance evaluation as seen from the above graph we can see that as the scale of the model was increasing the 
cuda event time was also increasing, this is because as the scale increases, the primitive size also increases and thus the min and max range in the 
for loop inside the rasterizer also increases and hence it takes more time to render. On the other habnd we also see that by doing back face culling we get a 
significant speed up as the number of  blocks lauched is decresed. Here thrust is used for stream compaction and when back face culling is performed
a bool is stored inside the primitive to tell if a primitive is facing towards the camera or away. Then by stream comapction the array size
is decreased. 


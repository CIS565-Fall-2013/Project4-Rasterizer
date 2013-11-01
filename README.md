-------------------------------------------------------------------------------
CIS565 Project 4: CUDA Rasterizer
===============================================================================
Ricky Arietta Fall 2013
-------------------------------------------------------------------------------

![Header] (https://raw.github.com/rarietta/Project4-Rasterizer/master/README_images/header.png)

-------------------------------------------------------------------------------
INTRODUCTION:
===============================================================================

This project implements a CUDA based implementation of a standard rasterized 
graphics pipeline, similar to the OpenGL pipeline. This includes CUDA based 
vertex shading, primitive assembly, perspective transformation, rasterization, 
fragment shading, and writing the resulting fragments to a framebuffer. 

-------------------------------------------------------------------------------
PART 1: 
===============================================================================
-------------------------------------------------------------------------------
Vertex Shader & Perspective Transformation
-------------------------------------------------------------------------------

Instead of having one VBO as is normal defined, my implementation had two VBO
arrays. The first VBO stored the vertex position floats in model space, and the
second stored them in projected screen space. The vertex shader received as 
arguments the un-transformed VBO, along with the three viewing matrices: the model
matrix from model to world space; the view matrix from world to camera space, and
the projection matrix from camera to screen space. Each vertex from the VBO would
be multiplied by these matrices, divided by the w coordinate to account for the
perspective divide, and transformed from unit screen space to the screen space
defined by the window height and width. 

-------------------------------------------------------------------------------
Primitive Assembly
-------------------------------------------------------------------------------

In this stage of the pipeline, I looked into the IBO to find each set of three
consecutive vertices that form a triangle primitive. I stored these three vertices
in the triangle struct in the primitives list at the correct index. However, I had
to modify the traditional triangle struct for my implementation. Rather than simply
storing 3 points and 3 colors, I changed the triangle to store 3 points and 3
fragments (each fragment having a position, color, and normal). The first three points
referred to the position vectors in screen space, while the fragment positions
were in the world frame of reference. This way, the rasterizer kernel could simply 
look into the triangle struct and determine the necessary values for interpolation 
at any point (explained below).

-------------------------------------------------------------------------------
Rasterization
-------------------------------------------------------------------------------

This is the crucial step in the pipeline. Taking in the primitives list, the rasterizer
kernel performed the scanline algorithm. This algorithm was parallelized by primitive,
so each kernel processed one triangle and rendered it to the screen.

Traditionally, with each fragment of a triangle primitive, the fragment would be
atomically compared to the depthbuffer to see if the primitive were occluded
or visible. UNFORTUNATELY, in my implementation, the atomic compare and exchange 
functions caused a CUDA timeout for any model other than tri.obj. As such, I have
commented the code out and my renders suffer from depth test issues. All other required
features, as well as the additional features, however, work perfectly fine.

-------------------------------------------------------------------------------
Lambertian Fragment Shading
-------------------------------------------------------------------------------

To achieve diffuse Lambertian shading within my fragment shader, I had to examine
the fragment stored at each pixel in the depthbuffer. For each fragment, I knew the
world-space position, the interpolated color, and the interpolated normal. From
this data, I could easily calculated a diffuse lighting coefficient according
to the following model:

![Diffuse] (https://raw.github.com/rarietta/Project4-Rasterizer/master/README_images/lambert.png)

This was combined with a hardcoded ambient coefficient to produce the following results.
The image on the right has the constant color value that is stored in the depthbuffer
for each fragment before the shader is called, while the image on the left
reflects the color value at each pixel after diffuse lighting calculations.

![Lambertian Fragment Shading] (https://raw.github.com/rarietta/Project4-Rasterizer/master/README_images/fragment_shader.png)

-------------------------------------------------------------------------------
Barycentric Color and Normal Interpolation
-------------------------------------------------------------------------------

To achieve proper color interpolation, I converted each pixel coordinate back to
barycentric coordinates relative to the triangle primitive's three vertices. Since
I already knew the linearly interpreted x- and y-position of each pixel relative
to the vertices' screen space positions, I could solve for the value of the
color as follows (image borrowed from UCSC presentation):

![Barycentric] (https://raw.github.com/rarietta/Project4-Rasterizer/master/README_images/barycentric.png)

By following this barycentric interpolation model for colors, I was able to
achieve properly interpolated color gradients within each face (in this example
I arbitrarily set the vertex colors to R, G, and B respectively):

![Color Interpolation] (https://raw.github.com/rarietta/Project4-Rasterizer/master/README_images/color_interpolation.png)

Furthermore, these barycentric coefficients could also be used for normal
interpolation at any point within the triangular face. The normal at any point
(x,y,z) could be calculated in exactly the same manner as a scaled sum of the
vertex normals. Below is a comparison of two rendered images of the cow. The one
on the left has constant normals per face, and the one on the right has smoothed vertex
normals with interpolated values for each (x,y,z):

![Normal Interpolation] (https://raw.github.com/rarietta/Project4-Rasterizer/master/README_images/normal_interpolation.png)

-------------------------------------------------------------------------------
Optimization: Back Face Culling
-------------------------------------------------------------------------------

To implement this optimization to the rasterizer, I examined each face within the
primitive assembly step. I determined whether or not the vertices of the face
were listed in clockwise or counterclockwise order in screen space. If they appear
in counterclockwise orientation, then the primitive is facing away from the camera
eye. In this case, the triangle's visibility field (which I added to the struct)
was marked false. When the triangle was sent through the rasterizing kernel,
the visibility field was checked before any calculations were performed and
any invisible primitive was ignored. Below is a video demonstrating this feature.
The triangle mesh has uniform normals initially facing towards the camera. As the
triangle is turned past 90 degrees, the normals begin to point away from the
camera and the triangle is no longer rendered.

[![Back-Faced Culling](https://raw.github.com/rarietta/Project4-Rasterizer/master/README_images/video_shot_2.png)] (https://raw.github.com/rarietta/Project4-Rasterizer/master/README_images/mousebased.avi)

An analysis of the benefits of the back face culling implementation on the
rasterizer at runtime is included as a performance analysis below.

-------------------------------------------------------------------------------
Mouse Based Camera Interactivity
-------------------------------------------------------------------------------

I also implemented mouse-based interactivity for the camera. By holding and dragging
the left mouse button, you can rotate the camera in any direction about the focal
point of the scene. By holding and dragging the right mouse button (or by holding
down SHIFT and the left mouse button again), you can zoom in or out. Below is a 
simple video demonstrating the mouse-based interactive camera feature:

[![Mouse-Based Interactive Camera](https://raw.github.com/rarietta/Project4-Rasterizer/master/README_images/video_shot_1.png)] (https://raw.github.com/rarietta/Project4-Rasterizer/master/README_images/mousebased.avi)

-------------------------------------------------------------------------------
PART 2: Performance Evaluation
===============================================================================

For this performance analysis, I ran FPS tests on two OBJ models both with and
without back face culling as described above. Judging from evaluation on the
included OBJ models (tri.obj, the simple triangle, and cow.obj, the above cow 
model), we observe a 30%-50% increase in runtime FPS with the addition of
back face culling. The data is found below in the graphs:

![Chart] (https://raw.github.com/rarietta/Project4-Rasterizer/master/README_images/chart.png)

![Graph] (https://raw.github.com/rarietta/Project4-Rasterizer/master/README_images/graph.png)

-------------------------------------------------------------------------------
NOTES:
===============================================================================

This project seeks to demonstrate how the graphics pipeline can be performed
with the use of a traditional raytracer. Therefore, no part of this rasterizer
uses any ray casting or tracing apart from the calculation of light direction 
in the Lambertian fragment shader.

Furthermore, no part of this pipeline utilizes preexisting OpenGL commands or
software. The only OpenGL found in this program occurs in drawing the Pixel Buffer
objects to the screen window, the pregeneration of camera matrices that would
otherwise be provided by the user, and the processing of mouse input for the
mouse based interactivity.

-------------------------------------------------------------------------------
ACKNOWLEDGMENTS:
===============================================================================

The basecode provided for this project by Patrick Cozzi and Liam Boone included 
an OBJ loader and much of the mundane I/O and bookkeeping code. The rest of the 
core rasterization pipeline, including all features discussed above, was 
implemented independently.

The only other outside code consulted in the implementation of this rasterizer
was meshview base code provided by Szymon Rusinkiewicz at Princeton University.
While the mouse-based interactivity was independently implemented for this project,
the Princeton base code was used as a reference for interacting with the hardware
via GLUT.

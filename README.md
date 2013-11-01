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

Vertex Shader Stuff

-------------------------------------------------------------------------------
Primitive Assembly
-------------------------------------------------------------------------------

Primitive Assembly Stuff

-------------------------------------------------------------------------------
Rasterization
-------------------------------------------------------------------------------

Rasterizer Stuff

-------------------------------------------------------------------------------
Lambertian Fragment Shading
-------------------------------------------------------------------------------

![Lambertian Fragment Shading] (https://raw.github.com/rarietta/Project4-Rasterizer/master/README_images/header.png)

-------------------------------------------------------------------------------
Barycentric Color and Normal Interpolation
-------------------------------------------------------------------------------

![Color Interpolation] (https://raw.github.com/rarietta/Project4-Rasterizer/master/README_images/color_interpolation.png)

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

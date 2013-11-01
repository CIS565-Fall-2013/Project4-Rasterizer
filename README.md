-------------------------------------------------------------------------------
CIS565 Project 4: CUDA Rasterizer
===============================================================================
Ricky Arietta Fall 2013
-------------------------------------------------------------------------------

![NBody Simulation] (https://raw.github.com/rarietta/Project4-Rasterizer/master/readme_images/normal_interpolation.png)

-------------------------------------------------------------------------------
INTRODUCTION:
===============================================================================

This project implements a CUDA based implementation of a standard rasterized 
graphics pipeline, similar to the OpenGL pipeline. This includes CUDA based 
vertex shading, primitive assembly, perspective transformation, rasterization, 
fragment shading, and writing the resulting fragments to a framebuffer. 

-------------------------------------------------------------------------------
PART 1: Vertex Shader & Perspective Transformation
===============================================================================

-------------------------------------------------------------------------------
PART 2: Primitive Assembly
===============================================================================

-------------------------------------------------------------------------------
PART 3: Rasterization
===============================================================================

-------------------------------------------------------------------------------
PART 4: Lambertian Fragment Shading
===============================================================================

Include why I added to fragment struct

-------------------------------------------------------------------------------
PART 5: Barycentric Color and Normal Interpolation
===============================================================================

-------------------------------------------------------------------------------
PART 6: Optimization: Back Face Culling
===============================================================================

-------------------------------------------------------------------------------
PART 7: Mouse Based Window Interactivity
===============================================================================

-------------------------------------------------------------------------------
PART 8: Performance Evaluation
===============================================================================
The performance evaluation is where you will investigate how to make your CUDA
programs more efficient using the skills you've learned in class. You must have
performed at least one experiment on your code to investigate the positive or
negative effects on performance. 

We encourage you to get creative with your tweaks. Consider places in your code
that could be considered bottlenecks and try to improve them. 

Each student should provide no more than a one page summary of their
optimizations along with tables and or graphs to visually explain any
performance differences.

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

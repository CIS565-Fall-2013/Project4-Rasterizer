-------------------------------------------------------------------------------
CUDA Based Rasterizer
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
NOTE:
-------------------------------------------------------------------------------
This project requires an NVIDIA graphics card with CUDA capability! Any card with CUDA compute capability 1.1 or higher will work fine for this project. For a full list of CUDA capable cards and their compute capability, please consult: http://developer.nvidia.com/cuda/cuda-gpus. If you do not have an NVIDIA graphics card in the machine you are working on, feel free to use any machine in the SIG Lab or in Moore100 labs. All machines in the SIG Lab and Moore100 are equipped with CUDA capable NVIDIA graphics cards. If this too proves to be a problem, please contact Patrick or Karl as soon as possible.


![Screenshot](/renders/ColorCube.JPG "Colored Cube")
![Screenshot](/renders/SmoothCowNormal.JPG "Smooth Cow")
![Screenshot](/renders/CowDepth.JPG "Cow Depth Buffer")

Youtube Video of Pipeline In Action
<dl>
<a href="http://youtu.be/fy26owMdrl4" target="_blank"><img src="http://img.youtube.com/vi/fy26owMdrl4/0.jpg" 
alt="Youtube Video of Rasterizer Running" width="640" height="480" border="10" /></a>
</dl>

-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
This is a simple OpenGL inspired graphics rasterization pipeline implemented in CUDA.

-------------------------------------------------------------------------------
CONTENTS:
-------------------------------------------------------------------------------
The Project4 root directory contains the following subdirectories:
	
* src/ contains the source code for the project. Both the Windows Visual Studio solution and the OSX makefile reference this folder for all source; the base source code compiles on OSX and Windows without modification.
* objs/ contains example obj test files: cow.obj, cube.obj, tri.obj.
* renders/ contains an example render of the given example cow.obj file with a z-depth fragment shader. 
* PROJ4_WIN/ contains a Windows Visual Studio 2010 project and all dependencies needed for building and running on Windows 7.

-------------------------------------------------------------------------------
Features
-------------------------------------------------------------------------------
This project implements the following stages of the graphics pipeline and features:
* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a naive scanline or a bin rasterizer
* Fragment Shading (Hot swappable between several precompiled programs)
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* Blinn-Phong shading, implemented in the fragment shader


In addition to these core pipeline functions, I've added this sugar:
   * Optional Back-face culling
   * Color/Normal interpolation across primitives
   * Mouse/Keyboard based interactive camera and view control
   * Alternative BIN RASTERIZER pipeline stage based on:
   	* High-Performance Software Rasterization on GPUs
		* Paper (HPG 2011): http://www.tml.tkk.fi/~samuli/publications/laine2011hpg_paper.pdf
		* Code: http://code.google.com/p/cudaraster/ Note that looking over this code for reference with regard to the paper is fine, but we most likely will not grant any requests to actually incorporate any of this code into your project.
		* Slides: http://bps11.idav.ucdavis.edu/talks/08-gpuSoftwareRasterLaineAndPantaleoni-BPS2011.pdf


-------------------------------------------------------------------------------
NOTES ON GLM:
-------------------------------------------------------------------------------
This project uses GLM, the GL Math library, for linear algebra. You need to know two important points on how GLM is used in this project:

* In this project, indices in GLM vectors (such as vec3, vec4), are accessed via swizzling. So, instead of v[0], v.x is used, and instead of v[1], v.y is used, and so on and so forth.
* GLM Matrix operations work fine on NVIDIA Fermi cards and later, but pre-Fermi cards do not play nice with GLM matrices. As such, in this project, GLM matrices are replaced with a custom matrix struct, called a cudaMat4, found in cudaMat4.h. A custom function for multiplying glm::vec4s and cudaMat4s is provided as multiplyMV() in intersections.h.


-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
Initally I implemented the rasterizer stage of the pipeline with a naive scan line method.
The rasterizer was primitive parallel and iterated over each primitive's Axis-Aligned Bounding Box (AABB).
For each fragment in the AABB, it would compute the mapped barycentric coordinates for the triangle.
If the coordniates generated were valid, the fragment was inside the triangle and proceeded to the depth test.

Because my implementation was primitive parallel, race conditions are possible in the depth buffer. 
To avoid this I used atomicCAS to enforce atomic writes.

Clearly there were many improvements that could be made. 

The first improvement was implementing backface culling. 
Instead of computing the dot product of the surface normal with the eye vector, I used the winding direction of the primitive in 2D screen space.
By ignoring the z component, this calculation reduces to 7 scalar operations.

The theoretical limit on performance improvement from backface culling is twofold. 
I suspected I would see about 75% of that due to overhead and the aditional pipeline stage.
As the charts below will show, I achieved roughly that on my naive implementation.

My second and more substantial improvement was the bin rasterizer. 
This splits the rasterize stage into a bin rasterizer, coarse rasterizer, and fine rasterizer.
Each stage splits the image plane into smaller subdivisions until the fine rasterizer can run completely pixel parallel.

The pipeline is much more complex, requiring multiple levels of intermediate buffers that must be maintained and accessed in parallel.
However, I was able to show a MASSIVE increase in performance, especially in the worst case scenario for the naive rasterizer.

This data was collected on a GeForce525M, Intel Core i5 laptop. 
This first chart shows a comparison between the different pipelines for the cube at various viewing angles and distances.
The x axis shows the average number of pixels per triangle. 
As the cube dominates the view screen, the bin rasterizer really shows it's worth.
![Screenshot](/performance/RasterizationRealtimeCube.JPG "Comparison of different methods for cube")
Also notice the substatial improvement the backface culling offers (especially for the naive implementation).

This result is robust for larger models such as the cow (5804 faces, 4583 vertices).
![Screenshot](/performance/RasterizationRealtimeCow.JPG "Comparison of different methods for cow")

Examining the application shows that the coarse rasterizer is now the bottleneck of the pipeline, accounting for >50% of total GPU Utilization in the worst cases.


-------------------------------------------------------------------------------
Additional Screenshots
-------------------------------------------------------------------------------
Flat Shaded Cow:
![Screenshot](/renders/CowFlatShading.JPG "Flat shaded cow")
Flat Cow Normals:
![Screenshot](/renders/FlatNormalCow.JPG "Cow Normals (Flat Cow)")
Depth Buffer For Cow:
![Screenshot](/renders/cow_zdepth.png "Cow Depth Buffer")
Cube:
![Screenshot](/renders/MatteCube.JPG "Simple Cube")
Depth Buffer for Cube:
![Screenshot](/renders/CubeDepth.JPG "Cube Depth Buffer")
Cube Normals:
![Screenshot](/renders/NormalCube.JPG "Cube Normals")


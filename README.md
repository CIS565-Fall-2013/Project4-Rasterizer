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



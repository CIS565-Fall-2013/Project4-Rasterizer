-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------
Qiong Wang
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
This is a course project based on CUDA for CIS 565 GPU Programming. The standard rasterization pipeline was implemented as the following steps:

1. Vertex Shader;
2. Primitive Assembly;
3. Rasterization(including basic back face culling);
4. Fragment Shader;
5. Rendering(including anti-aliasing)
6. Write to framebuffer

-------------------------------------------------------------------------------
IMPLEMENTED FEATURES:
-------------------------------------------------------------------------------
**Basic Graphics Pipeline and Features:**

* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline or a tiled approach
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple lighting/shading scheme with Lambert and Blinn-Phong in the fragment shader

**Additional Features:**

* Transformation feedback
* Back-face culling
* Atomic Compare and Swap for depth test
* Color interpretation between points on a primitive
* Anti-aliasing
* Both mouse-based and keyboard-based interactive camera support
* Multi-color lighting

-------------------------------------------------------------------------------
OPERATION INSTRUCTION
-------------------------------------------------------------------------------
In 22nd line of *rasterizeKernels.h*, the flag can be changed to 1 to use anti-aliasing mode as default.
``` cpp
#define ANTIALIASING 0
```

**Mouse Interaction**

|          Operation        |        Function      |
|:-------------------------:|:--------------------:|
| Left and right click drag | rotate around y-axis |
| Up and down click drag    | rotate around x-axis |

**Keyboard Interaction**

|          Operation        |            Function           |
|:-------------------------:|:-----------------------------:|
|          up-arrow         | zoom in along view direction  |
|          down-arrow       |zoom out along view direction  |
|          left-arrow       |  rotate from right to left    |
|          left-arrow       |  rotate from left to right    |
|             '['           |  rotate from down to up       |
|             ']'           |  rotate from up to down       |
|             'd'           |       depth mode or not       |
|             'f'           |   flat-color mode or not      |
|             '`'           |white color in flat-color mode |
|             '1'           | red color in flat-color mode  |
|             '2'           |green color in flat-color mode |
|             '3'           | blue color in flat-color mode |
|             '4'           |purple color in flat-color mode|

-------------------------------------------------------------------------------
SCREENSHOTS OF RESULTS
-------------------------------------------------------------------------------
* Depth in z-axis

![ScreenShot](https://raw.github.com/GabriellaQiong/Project4-Rasterizer/master/10302226.PNG)

* Transformation Feedback

![ScreenShot](https://raw.github.com/GabriellaQiong/Project4-Rasterizer/master/10302254.PNG)

* Color Interpolation

![ScreenShot](https://raw.github.com/GabriellaQiong/Project4-Rasterizer/master/10301538.PNG)

* Color Interpolation with Back Face Culling

![ScreenShot](https://raw.github.com/GabriellaQiong/Project4-Rasterizer/master/10311946.PNG)

* Flat Color with diffusion and specular reflection

![ScreenShot](https://raw.github.com/GabriellaQiong/Project4-Rasterizer/master/10311942.PNG)

![ScreenShot](https://raw.github.com/GabriellaQiong/Project4-Rasterizer/master/10311943.PNG)

![ScreenShot](https://raw.github.com/GabriellaQiong/Project4-Rasterizer/master/10311944.PNG)

![ScreenShot](https://raw.github.com/GabriellaQiong/Project4-Rasterizer/master/10311945.PNG)

* Anti-aliasing

Without anti-aliasing

![ScreenShot](https://raw.github.com/GabriellaQiong/Project4-Rasterizer/master/10311939.PNG)

With anti-aliasing

![ScreenShot](https://raw.github.com/GabriellaQiong/Project4-Rasterizer/master/10311940.PNG)

* Multi-color Lighting

![ScreenShot](https://raw.github.com/GabriellaQiong/Project4-Rasterizer/master/10311941.PNG)


-------------------------------------------------------------------------------
VIDEOS OF RESULTS
-------------------------------------------------------------------------------

This is the video of my CUDA rasterizer including mouse interaction and keyboard interaction

[![ScreenShot](https://raw.github.com/GabriellaQiong/Project4-Rasterizer/master/videoscreenshot.PNG)](http://www.youtube.com/watch?v=rb--TBxSOmw)

The youtube links are here if you cannot open the video in the markdown file: http://www.youtube.com/watch?v=rb--TBxSOmw

-------------------------------------------------------------------------------
PERFORMANCE EVALUATION
-------------------------------------------------------------------------------
Here is the table for the performance evaluation when adding different features to the rasterized cow.

|   Feature          |  approximate fps  |
|:------------------:|:-----------------:|
| Color Intepolation |       7~8         |
|    Z-depth         |       9~10        |
|   Anti-aliasing    |       5~6         |
|Multi-color Lighting|       4~5         |
| Back Face Culling  |       8~9         |

Note: the color interpolation without backface culling here is the default mode the z-depth mode and anti-aliasing and multi-color lighting all based on the color interpolation mode.

We can see that the back-face culling increase the frame per second a little bit while other features such as anti-aliasing, multi-color lighting all decrease the frame rate somehow.

-------------------------------------------------------------------------------
REFERENCES
-------------------------------------------------------------------------------
* Basic Pipeline of Rasterization:      http://cis565-fall-2013.github.io/lectures/10-14-Graphics-Pipeline.pptx
* Transformations from model to screen: http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/
* Phong-Blinn Specular Shading:         http://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model
* Atomic Compare and Swap:              https://devtalk.nvidia.com/default/topic/471383/atomiccas-doesn-39-t-compile-33-/
* AtomicCAS:                            http://en.wikipedia.org/wiki/Compare-and-swap

-------------------------------------------------------------------------------
ACKNOWLEDGEMENT
-------------------------------------------------------------------------------
Thanks a lot to Patrick and Liam for the preparation of this project. Thank you :)

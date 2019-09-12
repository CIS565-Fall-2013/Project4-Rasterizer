-------------------------------------------------------------------------------
CUDA Rasterizer
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
NOTE:
-------------------------------------------------------------------------------
This project requires an NVIDIA graphics card with CUDA capability! 
Any card with CUDA compute capability 1.1 or higher will work fine for this project.

-------------------------------------------------------------------------------
INTRODUCTION:
-------------------------------------------------------------------------------
This project is a CUDA based implementation of a simplified rasterization pipeline.

Very specifically, the following stages of the rasteriztion pipeline have been implemented,

- Vertex shader
- Primitive assembly
- Rasterization
- Fragment shader

-------------------------------------------------------------------------------
MY PIPELINE IMPLEMENTATION:
-------------------------------------------------------------------------------

![alt tag](https://raw.github.com/vimanyu/Project4-Rasterizer/master/doc_images/pipeline.png)

-------------------------------------------------------------------------------
FEATURES:
-------------------------------------------------------------------------------

* A basic rasterization pipeline from start to finish for obj models
* Parallel rasterization of each primitive
* Smooth interpolation of normals and colors using barycentric co-ordinates
* Backface culling in primitive assembly to discard back facing polygons. This is implemented using thrust's stream compaction.
* Interactive mouse movement for panning, zooming and rotating

-------------------------------------------------------------------------------
RESULTS:
-------------------------------------------------------------------------------
*COLOR INTERPOLATION* : Smooth linear interpolation of colors for each triangle based on barycentric co-ordinates
![alt tag](https://raw.github.com/vimanyu/Project4-Rasterizer/master/renders/cow_color_interp.png)

*NORMALS*: World space normals
![alt tag](https://raw.github.com/vimanyu/Project4-Rasterizer/master/renders/cow_normals.png)

*Z-DEPTH*: Z-depth in camera space
![alt tag](https://raw.github.com/vimanyu/Project4-Rasterizer/master/renders/dragon_depth.png)

*LAMBERT DIFFUSE LIGHING*: Simple lambertian shading
![alt tag](https://raw.github.com/vimanyu/Project4-Rasterizer/master/renders/dragon.png)

*BLINN PHONG LIGHTING*: Diffuse lighting with specular highlights
![alt tag](https://raw.github.com/vimanyu/Project4-Rasterizer/master/renders/buddha_spec.png)

-------------------------------------------------------------------------------
VIDEO
-------------------------------------------------------------------------------
The following is a video of the rasterizer tool in action

[![ScreenShot](https://raw.github.com/vimanyu/Project4-Rasterizer/master/doc_images/rasterizer_video_screenshot.png)](http://www.youtube.com/watch?v=s8ehsuIoL_U)

-------------------------------------------------------------------------------
BUILDING AND RUNNING CODE
-------------------------------------------------------------------------------
The code has been tested on Visual Studio 2012/2010 and cuda 5.5 on a laptop with compute capability 1.0 as well as 3.0.

Keyboard bindings for interactivity:

Key|Action
---|---
'c'| View color interpolation
'n'| View world space normals
'z'| View z-depth
'l'| Diffuse lighting
's'| Specular lighting
'r'| Reset camera

Mouse button| Action
LMB| Rotate 
MMB| Pan
RMB| Zoom

Apart from this you might encounter objs which have been built assuming z-axis as the up axis.
In this code, y-axis is the up-axis.

To correct this, you can simply, set the model matrix accordingly.

```
//main.cpp (line 96)
glm::vec3 myRotationAxis(1.0f, 0.0f, 0.0f);
//Change the 0.0f to -90f incase you need to align with y-axis
glm::mat4 rotationMat = glm::rotate( 0.0f, myRotationAxis )
```

If you need to render with a different color, you can alter,
```
//Control the number of bodies in the simulation
//rasterize_kernels.cu (line 345)
glm::vec3 col = glm::vec3(0.42f,0.35f,0.80f);
```
For now, these values are hard-coded but I will definitely be working on a future version that can read in a config file.

-------------------------------------------------------------------------------
PERFORMANCE ANALYSIS
-------------------------------------------------------------------------------
Performance tests were done on the following objs models

Model| Number of Vertices| Number of Faces
---|---|---
Bunny|4853|5804
Cow| 2503|4968
Budhda|49990| 100000
Dragon| 50000|100000


Backface culling:
Time is measured in *milliseconds* and measure the cuda kernel time per frame

Model|Without backface culling| Backfce culling during primitive assembly|Backface culling during fragment shader
---|---|---|---
Bunny|21.54|21.15| 19.36
Cow| 31.23|28.26|26.93
Budhda|50.37|43.67|36.68
Dragon| 45.63|40.95|39.18


The results seem to indicate that stream compaction is quite an overhead for models of this size.
Backface during primitive assembly is followed by stream compaction. This induces an overhead and maybe, we will see gains in more complex or bigger objs.

---
ACKNOWLEDGEMENTS
---
The objs for this project were downloaded from various websites online.

#*RasterMaster* (a.k.a. CIS565 Project 4: CUDA Rasterizer)

*RasterMaster* is a CUDA based implementation of a standard rasterized graphics pipeline, implementing  
the following stages:<br />
* Vertex shading<br />
* Primitive Assembly<br />
* Transformation and clipping<br />
* Rasterization<br />
* Fragment shading<br />
* Output merge/blend<br />
<br />

RasterMaster was developed as part of coursework at the University of Pennsylvania's 
[GPU Programming and Architecture (CIS 565)](http://www.seas.upenn.edu/~cis565) course. This project grew out of 
basecode provided by Yining Karl Li and Liam Boone which takes care of I/O and bookkeeping (such as loading an OBJ 
mesh, mathematical helper functions and so on). 

##Screenshots
<img src="https://raw.github.com/rohith10/RasterMaster/master/renders/OnePsychedelicCow.png" height="350" width="350"/><br />
<img src="https://raw.github.com/rohith10/RasterMaster/master/renders/WithOutlines.png" height="350" width="350"/><br />

##The Graphics Pipeline

The graphics rendering/rasterization pipeline varies slightly between OpenGL and Direct3D but roughly consists of the 
following steps:  
* Vertex shading: The vertex shader stage is mainly used for transforming a model from its own co-ordinate system 
(model space) to world space, camera space and finally to clip space (in that order). Clip space is the co-ordinate 
system the objects are in once perspective projection is applied. In this stage, the objects are contained within a truncated 
pyramid called the *view frustum*.<br />
* Primitive Assembly: In this stage, the vertices, normals and colours of an object, which are in separate arrays 
are grouped/assembled together based on the triangle these belong to. Hence the name "Primitive Assembly". <br />
* Transformation and clipping: In this stage, the models/meshes are transformed from the clip space to screen space. 
First, the view frustum that results from the perspective projection is transformed into a unit cube through an operation 
known as *Perspective divide*. At this stage, the x and y coordinates range from -1 to 1 and are called 
Normalized Device Coordinates (NDC). We convert these to screen-space coordinates specifying pixel values by first rescaling 
the NDC to the range [0, 1] and multiplying each with its respective resolution component. <br />
* Rasterization: In this stage we evaluate which pixels or fragments cover which object and set its colour accordingly. 
Since the fragment shader we use will not set any depth values for a pixel, we can perform early-Z test to set the right 
colours at each pixel.<br />
* Fragment shading: A shader program executes for every single pixel to compute the final shade due to lighting.<br />
* Output merge/blend: The rendered colour for a pixel is written to the screen after performing any sort of additive 
or multiplicative blending.

##Features

The following *required* stages and features of the graphics pipeline were implemented:

* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through either a scanline or a tiled approach
* Fragment Shading
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* A simple lighting/shading scheme, such as Lambert or Blinn-Phong, implemented in the fragment shader
* Back-face culling
* Correct color interpretation between points on a primitive
* Anti-aliasing

In addition, I also implemented the following *extra* features:
* MOUSE BASED interactive camera support. Although, this is glitchy in that the model of the cow is being displayed  
as a billboard or something. To view this feature, one must supply the following command line argument: "cameraControl=true".  
* Outlining all the triangles that make up the model. Press 'o' while running or set a command line parameter 
"outline=true", to view the model with such an effect.  

## Performance Evaluation
### Impact of Back-face culling on performance
Initially, I used a rasterization method that parallelized by pixel and was launched for every primitive. This was due to 
the fact that I was trying to work around having race conditions between threads in my kernel (I work on a Quadro FX 5600, 
which has compute capability 1.0 and therefore, lacks atomics).

Back-face culling improved rasterization performance on a linear scale as shown below:

<table>
<tr>
  <th>No. of triangles</th>
  <th>Run time (seconds)</th>       
</tr>
<tr>
  <td>~5800 (without back-face culling)</td>
  <td>52</td>
</tr>
<tr>
  <td>~2800 (with back-face culling)</td>
  <td>25</td>
</tr>
</table>

Model: Cow.obj  
  
However, I realized later that this was EXTREMELY inefficient and that it was indeed WAY better to parallelize by 
primitive. I switched to such a method and have been using that since.

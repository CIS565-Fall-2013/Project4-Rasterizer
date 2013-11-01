CIS 565 : Project 4 : CUDA Rasterizer

----

##Overview

We have implemented a GPU software rasterizer that takes into account the primary stages of the graphics pipeline.
This includes vertex shading, primitive assembly, viewport transformation, rasterizing and fragment shading.
We have been able to successfully a stanford dragon and a stanford bunny into our rasterizer, both rendering at ~20-30 FPS when optimized.
We hope to continue to optimize based on the observations we will discuss later on.

----

##Features

We have implemneted the following requirements:
* Vertex Shading
* Primitive Assembly
* Rasterization through a scanline approach and a tiled approach (both will be discussed later)
* Depth buffer for storing and depth testing
* Fragment to framebuffer writing
* Flat-shading & Lambert shading

and the following extras:
* Color Interpolation
* Mouse based interactive camera support
* Back-face ignoring / culling (with thrust)
* Non-geometric clipping
* Out-of-viewport ignoring

and debug modes:
* point mode
* normal rendering

Some of the following are planned / in-progress:
* mesh mode (front facing lines)
* per primitive scanline method that uses modified Bresenham

----

##Expected Results

#### Back-Face Culling
Most of the models that we see have half of the faces that are back-facing.  While we do not expect a 2x speedup,
we do expect to see the performance increase by a couple frames per second.  We also do think that, at one point in time
that back-face culling performance will drop below back-face ignoring, simply because of the overhead that is attributed
to running stream compaction on a large data set.

#### Non-Geometric Clipping
We will here-on refer to non-geometric clipping as out-of-viewport ignoring and force clipping at the rasterization stage. 
We propose that throwing out triangles that are completely out of the viewport after the viewport transformation stage
will help to improve performance later on, as there will be less triangles to put through the full rasterization stage.
We have also forcefully clipped the min and max of the bounding box to be at the viewport edge if any portion of the triangle
is outside the viewing port. We expect this to also improve performance by a bit as triangles grow larger and extend beyond
the viewport area.

----

##Performance Analysis

###Raw Data

##### Back-face Ignoring && Out-of-Viewport Ignoring
28 ms | 34 FPS

##### Back-face Culling && Out-of-Viewport Ignoring
30 ms | 32 FPS

 | 13 FPS
 
 
###Graphs

------

##Discussion

#### Rasterizing Method

#### Back-Face Culling 

#### Out-of-Viewport Ignoring

#### Non-Geometric Clipping

-----

## Acknowledgements

Many thanks to Ishaan who helped to implement point mode for debugging.

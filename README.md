#CIS 565 : Project 4 : CUDA Rasterizer

----

##Overview

We have implemented a GPU software rasterizer that takes into account the primary stages of the graphics pipeline.
This includes vertex shading, primitive assembly, viewport transformation, rasterizing and fragment shading.
We have been able to successfully a stanford dragon and a stanford bunny into our rasterizer, both rendering at ~20-30 FPS when optimized.
We hope to continue to optimize based on the observations we will discuss later on.

<div align="center">
<img src="https://raw.github.com/harmoli/Project4-Rasterizer/master/renders/dragon_normal.JPG" "Dragon with Shaded Normals">
<img src="https://raw.github.com/harmoli/Project4-Rasterizer/master/renders/bunny_color_diffuse.JPG" "Bunny with Color Interpolated Diffuse Shading">
</div>
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

##Mouse Interaction

* Left Button : Camera Rotate
* Middle Button : Camera Pan
* 'I' : Zoom In
* 'O' : Zoom Out

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

######Cow
Render | Total time | FPS
----- | ----- | -----
Back-face Ignoring | 18 ms | 51 FPS
Back-face Ignoring && Out-of-Viewport Ignoring |19-20 ms | 49 FPS
Back-face Culling && Out-of-Viewport Ignoring | 635 ms | 2 FPS
Back-face Culling | 620 ms | 2 FPS

######Stanford Bunny 
Render | Total time | FPS
----- | ----- | -----
Back-face Ignoring | 28 ms | 33 FPS
Back-face Ignoring && Out-of-Viewport Ignoring | 28 ms | 34 FPS
Back-face Culling && Out-of-Viewport Ignoring |30 ms | 32 FPS
Back-face Culling | 300 ms | 1 FPS
 
######Stanford Dragon
Render | Total time | FPS
----- | ----- | -----
Back-face Ignoring | 28 ms | 33 FPS
Back-face Ignoring && Out-of-Viewport Ignoring | 28 ms | 34 FPS
Back-face Culling && Out-of-Viewport Ignoring | 37 ms | 27 FPS
Back-face Culling | 36 ms | 26 FPS
 
###Graph

<div align="center">
<img src="https://docs.google.com/spreadsheet/oimg?key=0AgKtr6Wx5YDPdHBNQUdmSlRFTDg4bkdaWWhTb3JMN2c&oid=1&zx=gvuq1g9k5vva" />
<img src="https://docs.google.com/spreadsheet/oimg?key=0AgKtr6Wx5YDPdHBNQUdmSlRFTDg4bkdaWWhTb3JMN2c&oid=2&zx=u5yjmvff9j4s" />
</div>

------

##Discussion

#### Rasterizing Method
We investigated two methods of rasterizing : a per fragment/tiled method and a per primitive method.
It is no suprise that we found the former to perform much better with few large triangles, the very case where
the latter performs poorly.  Similarly, we found that the latter performs much better in large models that have many primitives.
This is beacuse the per fragment method is linear in performance in comparison to the number of primitives in the model
while the latter is linear in performance in comparison to the size of the triangles.  This means that the larger the triangle,
the worse the latter method performs.  Conversely, the more triangles in the model, the worse the former performs

The best way to remedy this would be to have a per primitive kernel that send off as many threads as each fragment that
exists in its bounding box.  However, the draw back of this method is the overhead that is introduced when many kernels
are started. We would expect that models with a large number of primitives that are small in size would not perform as 
well as those rasterized on a per primitive scanline basis.

It is also worth noting that we are currently using a Barycentric method that checks all pixels in the primitive's force-clipped
tight bounding box.  We could seek to improve rasterization performance slightly if we use a modified Bresenham algorithm
to fill the triangle starting at the calculated point the scanline intersects the left most edge.  This would avoid 
checking useless space without more overhead.

In all of the methods, we have chosen to leave out atomics as using atomics slows down the performance because warps run
in lock-step.  Thus, you can see some of the faces from the back peaking through from race conditions.  This is vastly helped
by back-face culling / ignoring.

<div align="center">
<img src="https://raw.github.com/harmoli/Project4-Rasterizer/master/renders/dragon_white.JPG" "Dragon White Diffuse w Race Conditions">
<img src="https://raw.github.com/harmoli/Project4-Rasterizer/master/renders/bunny_white.JPG" "Bunny White Diffuse, Race Conditions Unnoticeable">
</div>

#### Back-Face Culling 
Surprisingly, back-face culling is more expensive of an operation than back-face ignoring.  There is a possibility 
that thrust is to blame for this performance difference; however, it is probable that the overhead of performing stream
compaction outweighs the benefits of back-face culling altogether.  Similarly, back-face culling specifically requires 
2 more mallocs of large portions of memory equal to the number of primitives.  If the models are large, this memory 
is significant.

#### Non-Geometric Clipping
As seen here, we have seen slight improvements for non-geometric clipping.  This is mainly because we do not waste time
rasterizing useless large triangles that are outside of the viewport or not facing the user.  However, we would like to pose
a question : why is geometric clipping the standard?  While geometric clipping will ultimately have small triangles from 
frustum clipping and splitting the resulting geometry into triangles, does this not add a considerable amount of geometry
from the edges? If so, what are the biggest benefits that will come from geometric clipping?

-----

## Video

<iframe src="//player.vimeo.com/video/78317834" width="500" height="519" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe> <p><a href="http://vimeo.com/78317834">GPU Software Rasterizer</a> from <a href="http://vimeo.com/harmonyli">Harmony Li</a> on <a href="https://vimeo.com">Vimeo</a>.</p>

-----

## Acknowledgements

Many thanks to Ishaan who helped to implement point mode for debugging.

-----

## References

[OpenGL Transformation from World to Screen Space](http://www.songho.ca/opengl/gl_transform.html "OpenGL Transformations")

[OpenGL Tutorial on Matrices](http://www.opengl-tutorial.org/beginners-tutorials/tutorial-3-matrices/ "OpenGL Tutorial on Matrices")

[Triangle Rasterization Algorithms](http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html "Software Rasterization Triangle Algorithms")

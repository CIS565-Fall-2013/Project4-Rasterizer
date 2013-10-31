-------------------------------------------------------------------------------
CIS565: Project 4: CUDA Rasterizer
-------------------------------------------------------------------------------
Fall 2013
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
NOTE:
-------------------------------------------------------------------------------
NVidia Cards with compute 3.0 supported

---
Screenshots
---

*Diffuse Shading: 871k, 5k triangles*

![Screen](renders/dragon_diffuse.png)
![Screen](renders/cow_diffuse.png)

*Depth Shading: 100k triangles*

![Screen](renders/dragon.png)

*Normal Shading: 871k triangles*

![Screen](renders/dragon_Normal.png)

*Color Per Triangle: 6k triangles*

![Screen](renders/suzanne_col_per_tri.png)

*Color Per Vertex (interpolated Colors): 6k triangles*

![Screen](renders/suzanne_col_per_vert.png)

--- 
Features
---

Extra Features Impemented:
* Point Mode: Render using only points. Key 'p' to switch on and off
* Mouse Interaction: use mouse to pan and zoom
* Correct Color Interpolation: Using barycentric coordinates to interpolate colors
* Back Face Culling: Using thrust to remove faces with normals point away from the eye
* Back Face Ignoring: After calculating fragment properties, do not perform depth test in rasterization stage
* Dynamic Parallelization: use compute 3.5 to launch a kernel from within a kernel

Basic features implemented:
* Vertex Shading: Vertex transformations
* Primitive Assembly: Takes normal and color buffers and arranges them
* Rasterization: Per primitive rasterization
* Depth Testing: At rasterization stage
* Fragment shader: for different lighting techniques

---
Atomics
---

I tried implementing locking by using a mutex per fragment (storing it in the fragment buffer itself), but for some reason, it seemed to always be going to a deadlock situation.

---
Performance Analysis
---

One curious thing we realize is that this problem lends itself to dynamic parallelism since we don't want to write a loop over the pixels in the bounding box but rather do that parallely as well.

We notice that the per primitive rasterization stage is pixel bound. i.e., the limit depends on the screen-space size of the triangle being rasterized since we have one triangle per thread. We could potentially parallelize by thread per fragment in output fragment buffer but then we would be bound by the number of fragments in the scene instead.
Thus, this implementation works most efficiently if there are small triangles. One large triangle covering most of the space would be a bad case scenario where this implementation doesn't scale down well.

Some small optimizations that I ended up doing were the following:
* Ensuring that the bounding box of the triangle is clipped in the viewport space.
* Not rasterizing on the line in case the current primitive has been crossed twice. In the worst case, this doesn't offer any speed up, in the best case, this speeds up one thread by 2x.

None the less, because the above optimizations are divergent, the speedups do not linearly translate to lowering times.

Another interesting artifact noticed was that the back face culling seems to be faster than back face ignoring. I think this can be explained by the re-mallocing internally by thrust's remove-if while in back face ignoring, we just ignore the entire raster step for that triangle but it still exists in the list. The speed up is about 1-2 fps for medium scale models O(10k).
I think a better metric would be to compare the same by varying the screen space average size of the triangle.
CIS565: Project 4: CUDA Rasterizer
===
Fall 2013
---
Yingting Xiao
---

![alt tag](https://raw.github.com/YingtingXiao/Project4-Rasterizer/master/renders/bunny.PNG)

![alt tag](https://raw.github.com/YingtingXiao/Project4-Rasterizer/master/renders/dragon.PNG)

---
Requirements completed
---

* Vertex Shading
* Primitive Assembly with support for triangle VBOs/IBOs
* Perspective Transformation
* Rasterization through scanline
* Fragment Sading + per-fragment lighting
* A depth buffer for storing and depth testing fragments
* Fragment to framebuffer writing
* Lamber shading

---
Additional features
---

* Color interpolation

![alt tag](https://raw.github.com/YingtingXiao/Project4-Rasterizer/master/renders/color_interpolation.PNG)

* Camera interaction

Left button - rotate

Middle button - pan

Right button (move horizontally) - zoom

Demo: https://vimeo.com/78330880

* Back-face culling

See analysis below

---
Performance Analysis
---

![alt tag](https://raw.github.com/YingtingXiao/Project4-Rasterizer/master/perf.PNG)

I did rasterization through scanline-parallel scanline approach. This is more efficient than the pixel-parallel approach without bounding box. since I just need to test intersection with all the premitives for every scanline instead of every pixel. However, compared to the primitive-parallel approach with bounding box (only tests intersection between primitives and the pixels around them), my approach is certainly not efficient enough.

Surprisingly, the perfomance of my rasterizer declined after I added back-face culling. I added back-face culling to my rasterizer kernel. I checked if the dot products of a primitive's normals and the view direction are all greater than 1. If so, the primitive is facing away from the camera. Therefore, we do not test if it intersects with the scanline. Since I didn't use stream compaction, back-face culling should only shorten some threads' running time, and therefore should not cause a huge improvement on the overall running time. I also added 3 * numberOfPrimitives dot product computation to each thread. I think this is the reason that back-face culling slows down my program.

---
Future work
---

There are a lot of optimizations that could be done. I want to do primitive-parallel scanline, which only checks for intersections between a primitive and the scanlines that are nearby. I believe that this will improve the performance significantly. I also want to use stream compaction together with back-face culling and scissor test.
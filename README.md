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
* Lambert shading

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

* Back-face culling and scissor test

See analysis below

---
Performance Analysis
---

![alt tag](https://raw.github.com/YingtingXiao/Project4-Rasterizer/master/perf.PNG)

I first did rasterization through scanline-parallel scanline approach. Each thread checks a scanline's intersection with all primitives and fill in the fragments that intersect with primitives. This is more efficient than the pixel-parallel approach without bounding box, since I just need to test intersection with all the premitives for every scanline instead of every pixel. However, compared to the primitive-parallel approach with bounding box (only tests intersection between primitives and the pixels around them), my approach is certainly not efficient enough. So I tried primitive-parallel scanline approach, where I look for intersection between primitives and their nearby scanlines. This enhanced my program's performance significantly, as shown in the chart above. On my GT650M GPU, my rasterizer runs at 40fps with the stanford bunny.

Then I did backface culling and clipping with self-written stream compaction. Surprisingly, the perfomance of my rasterizer went down. I think this is due to the heavy computation and memory allocation used in stream compaction, so I optimized my stream compaction by allocating memory only when I initialize CUDA. This brought my speed back to the speed before I did any backface culling and clipping, which is not very encouraging :( I also tried doing backface culling and clipping without stream compaction (i.e. as an if statement in rasterizeKernel). This also slowed down my program, but not as much as unoptimized stream compaction. However, clipping does have a significant effect on performace when more than half of the object is moved out of screen (as shown in the demo).

---
Future work
---

Some pixels are flicking due to multiple threads accessing the same depth buffer. I want to eliminate this by locking the memory, as suggested by Hao in the Google group.

I also really want to implement this, once I figure out how...

https://www.youtube.com/watch?v=5DKIP9N-OB4
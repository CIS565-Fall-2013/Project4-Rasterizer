CUDA Rasterizer in Progress
Credits: 
1. I got the idea for fAtomicMax from the NVIDIA forums: https://devtalk.nvidia.com/default/topic/492068/atomicmin-with-float/, however instead of copying it, I copied atomicAdd from the slides and turned it into my own fAtomicMax, without looking at the forum post. It probably came out almost identical to the forum post anyway, since that forum post is also based on the example that the slide is based on. 
2. Some snippets from http://sol.gfxile.net/tri/index.html. The sort I did myself without looking at the code there. The gradient calculation I did steal (but that's basically the only way to calculate gradient, and it's also three lines). The pseudocode/algorithm description is what I did, but that's something I did myself, since a full implentation isn't on that blog post.

Known Issues:
Zooming the camera inside the body causes a crash.

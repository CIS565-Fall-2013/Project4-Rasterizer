clear all
close all
importfile('cow_bin.csv');
cow_bin = packstruct(AvgPixelsPerTri,MaxPixelsPerTri,NumTriangles, NumTrianglesRastered, rasterTimeSecs);
importfile('cow_naive.csv');
cow_naive = packstruct(AvgPixelsPerTri,MaxPixelsPerTri,NumTriangles, NumTrianglesRastered, rasterTimeSecs);
importfile('cow_bin_culling.csv');
cow_bin_culling = packstruct(AvgPixelsPerTri,MaxPixelsPerTri,NumTriangles, NumTrianglesRastered, rasterTimeSecs);
importfile('cow_naive_culling.csv');
cow_naive_culling = packstruct(AvgPixelsPerTri,MaxPixelsPerTri,NumTriangles, NumTrianglesRastered, rasterTimeSecs);

importfile('cube-bin.csv');
cube_bin = packstruct(AvgPixelsPerTri,MaxPixelsPerTri,NumTriangles, NumTrianglesRastered, rasterTimeSecs);
importfile('cube-naive.csv');
cube_naive = packstruct(AvgPixelsPerTri,MaxPixelsPerTri,NumTriangles, NumTrianglesRastered, rasterTimeSecs);
importfile('cube-bin-culling.csv');
cube_bin_culling = packstruct(AvgPixelsPerTri,MaxPixelsPerTri,NumTriangles, NumTrianglesRastered, rasterTimeSecs);
importfile('cube-naive-culling.csv');
cube_naive_culling = packstruct(AvgPixelsPerTri,MaxPixelsPerTri,NumTriangles, NumTrianglesRastered, rasterTimeSecs);




figure
scatter(cow_bin.AvgPixelsPerTri, cow_bin.rasterTimeSecs*1000, '+');
hold all
scatter(cow_naive.AvgPixelsPerTri, cow_naive.rasterTimeSecs*1000, '.');
scatter(cow_bin_culling.AvgPixelsPerTri, cow_bin_culling.rasterTimeSecs*1000, '+');
scatter(cow_naive_culling.AvgPixelsPerTri, cow_naive_culling.rasterTimeSecs*1000, '.');
plot([-1000 10000], [16 16]);
hold off
ylabel('Rasterization Time (ms)')
xlabel('Pixels');
title('Avg Pixels per Triangle vs Render time (Cow)');
legend('Bin Raster', 'Naive Raster', 'Bin Raster w/ BF Culling', 'Naive Raster w/ BF Culling', '16ms')
axis([0 1000 0 150])



figure
scatter(cube_bin.AvgPixelsPerTri, cube_bin.rasterTimeSecs*1000, '+');
hold all
scatter(cube_naive.AvgPixelsPerTri, cube_naive.rasterTimeSecs*1000, '.');
scatter(cube_bin_culling.AvgPixelsPerTri, cube_bin_culling.rasterTimeSecs*1000, '+');
scatter(cube_naive_culling.AvgPixelsPerTri, cube_naive_culling.rasterTimeSecs*1000, '.');
plot([-1000 10000], [16 16]);
hold off
ylabel('Rasterization Time (ms)')
xlabel('Pixels');
title('Avg Pixels per Triangle vs Render time (Cube)');
legend('Bin Raster', 'Naive Raster', 'Bin Raster w/ BF Culling', 'Naive Raster w/ BF Culling', '16ms')
axis([0 3000 0 50])



figure
scatter(cube_bin.AvgPixelsPerTri, cube_bin.rasterTimeSecs*1000000./cube_bin.NumTrianglesRastered, '+');
hold all
scatter(cube_naive.AvgPixelsPerTri, cube_naive.rasterTimeSecs*1000000./cube_naive.NumTrianglesRastered, '.');
scatter(cube_bin_culling.AvgPixelsPerTri, cube_bin_culling.rasterTimeSecs*1000000./cube_bin_culling.NumTrianglesRastered, '+');
scatter(cube_naive_culling.AvgPixelsPerTri, cube_naive_culling.rasterTimeSecs*1000000./cube_naive_culling.NumTrianglesRastered, '.');

scatter(cow_bin.AvgPixelsPerTri, cow_bin.rasterTimeSecs*1000000./cow_bin.NumTrianglesRastered, 'o');
scatter(cow_naive.AvgPixelsPerTri, cow_naive.rasterTimeSecs*1000000./cow_naive.NumTrianglesRastered, '*');
scatter(cow_bin_culling.AvgPixelsPerTri, cow_bin_culling.rasterTimeSecs*1000000./cow_bin_culling.NumTrianglesRastered, 'o');
scatter(cow_naive_culling.AvgPixelsPerTri, cow_naive_culling.rasterTimeSecs*1000000./cow_naive_culling.NumTrianglesRastered, '*');
hold off
ylabel('Render Time Per Triangle(ns)')
xlabel('Pixels');
title('Avg Pixels per Triangle vs Render time (Cube)');
legend('Cube Bin Raster', 'Cube Naive Raster', 'Cube Bin Raster w/ BF Culling', 'Cube Naive Raster w/ BF Culling', 'Cow Bin Raster', 'Cow Naive Raster', 'Cow Bin Raster w/ BF Culling', 'Cow Naive Raster w/ BF Culling')
%axis([0 1000 0 150])
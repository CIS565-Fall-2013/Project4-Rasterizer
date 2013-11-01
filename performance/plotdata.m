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



figure
scatter(cow_bin.AvgPixelsPerTri, cow_bin.rasterTimeSecs*1000, '+');
hold all
scatter(cow_naive.AvgPixelsPerTri, cow_naive.rasterTimeSecs*1000, '.');
scatter(cow_bin_culling.AvgPixelsPerTri, cow_bin_culling.rasterTimeSecs*1000, '+');
scatter(cow_naive_culling.AvgPixelsPerTri, cow_naive_culling.rasterTimeSecs*1000, '.');
hold off
ylabel('Rasterization Time (ms)')
xlabel('Pixels');
title('Avg Pixels per Triangle vs Render time');
legend('Bin Raster', 'Naive Raster', 'Bin Raster w/ BF Culling', 'Naive Raster w/ BF Culling')
axis([0 1000 0 150])


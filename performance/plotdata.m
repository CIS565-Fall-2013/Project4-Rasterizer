clear all
%close all
importfile('cow_bin.csv');
cow_bin = packstruct(AvgPixelsPerTri,MaxPixelsPerTri,NumTriangles, NumTrianglesRastered, rasterTimeSecs);
importfile('cow_naive.csv');
cow_naive = packstruct(AvgPixelsPerTri,MaxPixelsPerTri,NumTriangles, NumTrianglesRastered, rasterTimeSecs);

figure
scatter(cow_bin.AvgPixelsPerTri, cow_bin.rasterTimeSecs*1000, '.');
hold all
scatter(cow_naive.AvgPixelsPerTri, cow_naive.rasterTimeSecs*1000, '.');
hold off
ylabel('Rasterization Time (ms)')
xlabel('Pixels');
title('Avg Pixels per Triangle vs Render time. No Culling');
legend('Bin Raster', 'Naive Raster')



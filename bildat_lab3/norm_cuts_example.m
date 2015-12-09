colour_bandwidth = 100.0; % color bandwidth
%radius = 3;              % maximum neighbourhood distance
%ncuts_thresh = 0.2;      % cutting threshold
%min_area = 200;          % minimum area of segment
%max_depth = 8;           % maximum splitting depth
scale_factor = 0.4;      % image downscale factor
image_sigma = 2.0;       % image preblurring scale

radius = 3;              % maximum neighbourhood distance
ncuts_thresh = 0.5;      % cutting threshold
min_area = 10;         % minimum area of segment
max_depth = 8;           % maximum splitting depth



I = imread('orange.jpg');
%I = imread('tiger1.jpg');
%I = imread('tiger2.jpg');
%I = imread('tiger3.jpg');


I = imresize(I, scale_factor);
Iback = I;
d = 2*ceil(image_sigma*2) + 1;
h = fspecial('gaussian', [d d], image_sigma);
I = imfilter(I, h);

segm = norm_cuts_segm(I, colour_bandwidth, radius, ncuts_thresh, min_area, max_depth);
Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);

dest1 = strcat('result/normcuts1', '_ma_', num2str(min_area), ...
    '_nt_', num2str(ncuts_thresh), '_md_', num2str(max_depth), '.png');

dest2 = strcat('result/normcuts2', '_ma_', num2str(min_area), ...
    '_nt_', num2str(ncuts_thresh), '_md_', num2str(max_depth), '.png');

imwrite(Inew, dest1)
imwrite(I, dest2)

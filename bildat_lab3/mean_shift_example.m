scale_factor = 1.0;       % image downscale factor
image_sigma = 1.0;        % image preblurring scale

num_iterations = 40;      % number of mean-shift iterations

sb = [8];
cb = [1.0 1.2 1.4 1.6 1.8 2.0];

for i=1:size(sb,2)
    for j=1:size(cb,2)
        %spatial_bandwidth = 2.0; % spatial bandwidth
        %colour_bandwidth = 10.0;   % colour bandwidth
        
        spatial_bandwidth = sb(i);
        colour_bandwidth = cb(j);
        
        %I = imread('orange.jpg');
        %I = imread('tiger1.jpg');
        %I = imread('tiger2.jpg');
        I = imread('tiger3.jpg');
        I = imresize(I, scale_factor);
        Iback = I;
        d = 2*ceil(image_sigma*2) + 1;
        h = fspecial('gaussian', [d d], image_sigma);
        I = imfilter(I, h);

        segm = mean_shift_segm(I, spatial_bandwidth, colour_bandwidth, num_iterations);
        Inew = mean_segments(Iback, segm);
        I = overlay_bounds(Iback, segm);

        dest1 = strcat('result/meanshift1_', num2str(spatial_bandwidth), '_', num2str(colour_bandwidth), '.png');
        dest2 = strcat('result/meanshift2_', num2str(spatial_bandwidth), '_', num2str(colour_bandwidth), '.png');

        imwrite(Inew, dest1)
        imwrite(I, dest2)
        subplot(1,2,1); imshow(Inew);
        subplot(1,2,2); imshow(I);
    end
end
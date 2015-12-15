%scale_factor = 0.5;          % image downscale factor
%area = [ 80, 110, 570, 300 ] % image region to train foreground with
%K = 16;                      % number of mixture components
%alpha = 8.0;                 % maximum edge cost
%sigma = 10.0;                % edge cost decay factor

scale_factor = 0.5;           % image downscale factor
area = [ 80, 110, 570, 300 ]; % image region to train foreground with
K = 15;                       % number of mixture components

alpha = 8.0;                  % maximum edge cost
sigma = 10.0;                 % edge cost decay factor


area = [ 80, 110, 570, 300 ]; % image region to train foreground with
I = imread('tiger1.jpg');
I = imresize(I, scale_factor);
Iback = I;
area = int16(area*scale_factor);
[ segm, prior ] = graphcut_segm(I, area, K, alphas(a), sigmas(s));

Inew = mean_segments(Iback, segm);
I = overlay_bounds(Iback, segm);

g1 = strcat('result/graphcut1_k_', num2str(K), '.png');
g2 = strcat('result/graphcut2_k_', num2str(K), '.png');
g3 = strcat('result/graphcut3_k_', num2str(K), '.png');

imwrite(Inew, g1)
imwrite(I, g2)
imwrite(prior, g3)

subplot(2,2,1); imshow(Inew);
subplot(2,2,2); imshow(I);
subplot(2,2,3); imshow(prior);


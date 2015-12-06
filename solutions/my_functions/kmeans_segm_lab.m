% Given an image,
% the number of cluster centers K,
% number of iterations L and
% a seed for initializing randomization,
% computes a segmentation (with a colour index per pixel) and the centers of all
% clusters in 3D colour space.
function [ segmentation, centers ] = kmeans_segm(image, K, L, seed)

  % Convert to L*a*b
  cform = makecform('srgb2lab');
  lab = applycform(image, cform);

  % Cast to double for the necessary computations
  lab_double = im2double(lab);

  % The height and width of the image
  height = size(lab_double, 1);
  width = size(lab_double, 2);

  % Reshape into 2D taking only the a* and b* components
  lab_2D = reshape(lab_double, height * width, 3);
  ab_2D = reshape(lab_double(:,:,2:3), height * width, 2);


  % Initialize random centers

  centers = [0.00526049995830391 -0.010408184525267927;
    29.665796908927955 76.09560152548369];

  rng(seed);
  rand_colours = 256 .* rand(K, 2) - 128;

  centers = [centers; rand_colours];

  K = size(centers, 1);


  % This maps every pixel to the kernel closest to it
  kernel_with_min_distance = zeros(height * width, 1);

  % A pixels x centers matrix.
  pixel_to_kernel_distance = zeros(height * width, K);


  for l = 1:L
    % --------------------------------------------------------------------------
    % Assign each pixel to the cluster center for which the distance is minimum
    pixel_to_kernel_distance  = pdist2(ab_2D, centers, 'euclidean');

    [min_distance kernel_with_min_distance] = min(pixel_to_kernel_distance, [], 2);


    % --------------------------------------------------------------------------
    % Recompute each cluster center by taking the
    % mean of all pixels assigned to it
    for kernel = 1:K

      pixels = 0;

      means = zeros(2,1);
      means = double(means);

      for pixel = 1:height * width
        if kernel_with_min_distance(pixel) == kernel

          pixels = pixels + 1;

          means(1) = ((pixels - 1) * means(1) + ab_2D(pixel, 1)) / pixels;
          means(2) = ((pixels - 1) * means(2) + ab_2D(pixel, 2)) / pixels;
        end
      end

      centers(kernel, :) = means;

    end % End kernel loop

  end % End l loop


  segmentation = zeros(height * width, 3);

  segmentation(:,1) = lab_2D(:,1);
  segmentation(:,2) = centers(kernel_with_min_distance(:,1), 1);
  segmentation(:,3) = centers(kernel_with_min_distance(:,1), 2);

  segmentation = reshape(segmentation, height, width, 3);

  % Convert to RGB
  cform = makecform('lab2srgb');
  segmentation = applycform(segmentation, cform);

  segmentation = segmentation .* 255;

  imshow(segmentation);

end % End function

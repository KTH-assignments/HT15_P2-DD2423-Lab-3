function prob = mixture_prob(image, K, L, mask)

  % The height and width of the image
  [height, width, ~] = size(image);

  HW = height * width;

  % Reshape image into 2D
  image_vec = single(reshape(image, width * height, 3));

  % ------------ Store all pixels for which mask=1 in a Nx3 matrix -------------
  % Identify the indices of the non-zero pixels
  mask_non_zeros = find(mask);

  % The image masked
  image_masked = image .* repmat(mask, [1 1 3]);

  % The masked image in 2D
  image_masked_vec = im2double(reshape(image_masked, HW, 3));

  % --------- Randomly initialize the K components using masked pixels ---------
  % mu_k -> centers
  [segmentation centers] = kmeans_segm(image_masked, K, L, 1421);

  % Reshape segmentation into one column
  segmentation = reshape(segmentation, size(segmentation,1) * size(segmentation,2), 1);

  % sigma_k
  cov = cell(K,1);
  cov(:) = {eye(3)};

  % w_k
  w = zeros(K, 1);

  for i = 1:K
    w(i) = sum(segmentation(:,1) == i);
  end


  % Initialize p(i,k) for the masked pixels
  p = zeros(HW, K);

  % Initialize g(i,k) for the masked pixels
  g = zeros(HW, K);

  % The nominator of Step 1
  nom = zeros(HW, K);

  % Iterate L times
  for l = 1:2

    % The denominator of Step 1
    sum_denom = zeros(HW, 1);

    % ------ Expectation: Compute probabilities P_ik using masked pixels -------
    for kernel = 1:K

      diff = bsxfun(@minus, image_masked_vec, centers(kernel, :));

      g(:,kernel) = ((2 * pi)^3 * det(cov{kernel}))^(-1/2) * ...
        exp(-0.5 * sum(diff * inv(cov{kernel}) .* diff, 2));

      nom(:,kernel) = w(kernel) * g(:, kernel);

      sum_denom = sum_denom + nom(:,kernel);
    end

    for kernel = 1:K
      p(:,kernel) = nom(:,kernel) .* spfun(@(x) 1./x, sum_denom);
    end


    % Maximization: Update weights, means and covariances using masked pixels --
    for kernel = 1:K
      w(kernel) = sum(p(:, kernel), 1) / HW;

      if sum(p(:, kernel), 1) ~= 0
        centers(kernel, :) = p(:,kernel)' * image_masked_vec / sum(p(:, kernel), 1);
      end
    end


    for kernel = 1:K

      diff = bsxfun(@minus, image_masked_vec, centers(kernel, :));

      if sum(p(:, kernel), 1) ~= 0
        cov{kernel} = (sum(p(:, kernel)' * diff * diff') / sum(p(:, kernel), 1)) * eye(3);
      end

    end

  end

  prob = reshape(g*w, height, width, 1);

end

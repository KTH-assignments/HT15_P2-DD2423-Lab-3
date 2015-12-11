function prob = mixture_prob(image, K, L, mask)

  % The height and width of the image
  [height, width, ~] = size(image);

  % Reshape image into 2D
  image_vec = single(reshape(image, width * height, 3));

  % ------------ Store all pixels for which mask=1 in a Nx3 matrix -------------
  % Identify the indices of the non-zero pixels
  mask_non_zeros = find(mask);

  % The size of the mask
  N = size(mask_non_zeros, 1);

  % The mask translated into the image
  image_masked = image_vec(mask_non_zeros, :, :);

  image_masked_resh = reshape(image_masked, N, 1, 3);

  % --------- Randomly initialize the K components using masked pixels ---------
  % mu_k -> centers
  [segmentation centers] = kmeans_segm(image_masked_resh, K, L, 1421);

  % sigma_k
  cov = cell(K,1);
  cov(:) = {eye(3)};

  % w_k
  w = zeros(K, 1);

  for i = 1:K
    w(i) = sum(segmentation(:,1) == i);
  end

  % Initialize p(i,k) for the masked pixels
  p = zeros(N, K);

  % Initialize g(i,k) for the masked pixels
  g = zeros(N, K);

  % Iterate L times
  for l = 1:L

    % The denominator of Step 1
    sum_denom = zeros(N, 1);

    % The nominator of Step 1
    nom = zeros(N, K);

    % ------ Expectation: Compute probabilities P_ik using masked pixels -------
    for kernel = 1:K
      diff = bsxfun(@minus, image_masked, centers(kernel, :));

      g(:,kernel) = ((2 * pi)^3 * det(cov{kernel}))^(-1/2) * ...
        exp(-0.5 * sum(diff * inv(cov{kernel}) .* diff, 2));

      nom(:,kernel) = w(kernel) * g(:, kernel);

      sum_denom = sum_denom + nom(:,kernel);
    end

    for kernel = 1:K
      p(:,kernel) = nom(:,kernel) ./ sum_denom;
    end


    % Maximization: Update weights, means and covariances using masked pixels --
    for kernel = 1:K
      w(kernel) = sum(p(:, kernel), 1) / N;

      centers(kernel, :) = p(:,kernel)' * image_masked / sum(p(:, kernel), 1);
    end

    for kernel = 1:K
      diff = bsxfun(@minus, image_masked, centers(kernel, :));
      cov{kernel} = sum(p(:, kernel)' * diff * diff') / sum(p(:, kernel), 1) * eye(3);
    end
  end

end

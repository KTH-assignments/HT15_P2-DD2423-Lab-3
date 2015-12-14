function prob = mixture_prob(image, K, L, mask)

  % The height and width of the image
  [height, width, ~] = size(image);

  N = height * width;

  % Reshape image into 2D
  image_vec = single(reshape(image, width * height, 3));

% -------------- Store all pixels for which mask=1 in a Nx3 matrix -------------

  % The image masked
  image_masked = image .* repmat(mask, [1 1 3]);

  % The masked image in 2D
  image_masked_vec = im2double(reshape(image_masked, N, 3));


% ----------- Randomly initialize the K components using masked pixels ---------

  % mu_k -> centers
  [segmentation centers] = kmeans_segm(image_masked, K, L, 1421);

  % Reshape segmentation into one column
  segmentation = reshape(segmentation, size(segmentation,1) * size(segmentation,2), 1);

  % sigma_k
  cov = cell(K,1);
  cov(:) = {rand * 10000 * eye(3) + ones(3,3)};

  % w_k
  w = zeros(K, 1);

  for i = 1:K
    w(i) = sum(segmentation(:,1) == i) + 1;
  end

  w = w / sum(w,1);


  % Initialize p(i,k) for the masked pixels
  p = zeros(N, K);

  % Initialize g(i,k) for the masked pixels
  g = zeros(N, K);

  % The nominator of Step 1
  nom_p = zeros(N, K);


% ------------------------------- Iterate L times ------------------------------
  for l = 1:L

% --------- Expectation: Compute probabilities P_ik using masked pixels --------
    for kernel = 1:K

      diff = abs(bsxfun(@minus, image_masked_vec, centers(kernel, :)));

      g(:,kernel) = ((2 * pi)^3 * abs(det(cov{kernel})))^(-1/2) * ...
        exp(-0.5 * sum(diff * inv(cov{kernel}) .* diff, 2));

      % g should sum up to one
      g(:,kernel) = g(:,kernel) / sum(g(:,kernel), 1);

      nom_p(:,kernel) = w(kernel) * g(:, kernel);

    end


    % The denominator of Step 1
    sum_denom_p = sum(nom_p,2);


    for kernel = 1:K
      p(:,kernel) = nom_p(:,kernel) ./ sum_denom_p;
    end


% --- Maximization: Update weights, means and covariances using masked pixels --
    for kernel = 1:K
      w(kernel) = sum(p(:, kernel), 1) / N;

      centers(kernel, :) = p(:,kernel)' * image_masked_vec / (sum(p(:, kernel), 1));
    end

    % Normalize the weights
    w = w / sum(w,1);

    for kernel = 1:K

      diff = bsxfun(@minus, image_masked_vec, centers(kernel, :));

      d = diff' * (diff .* repmat(p(:,kernel), [1 3]));

      diag_d = diag(diag(d));
      cov{kernel} =  (diag_d) / sum(p(:, kernel), 1);

    end

  end

  k = (g * w);
  k = k / sum(k,1);


  prob = reshape(k, height, width, 1);

end

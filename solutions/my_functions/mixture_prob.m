function prob = mixture_prob(image, K, L, mask)

  % The height and width of the image
  [height, width, ~] = size(image);

  N = height * width;


% -------------- Store all pixels for which mask=1 in a Nx3 matrix -------------

  % Reshape image into 2D
  image_vec = im2double(reshape(image, N, 3));

  % Identify and obtain the masked pixels
  image_masked_vec = image_vec(find(mask), :);

  % The total number of masked pixels
  M = size(image_masked_vec, 1);


% ----------- Randomly initialize the K components using masked pixels ---------

  % mu_k -> centers
  [segmentation centers] = kmeans_segm_for_mixture(image_masked_vec, K, 10*L, 1421);

  % sigma_k
  cov = cell(K,1);
  cov(:) = {0.04 * rand * eye(3)};

  % w_k
  w = zeros(K, 1);

  for i = 1:K
    w(i) = sum(segmentation(:,1) == i);
  end

  w = w / sum(w,1);


  % Initialize p(i,k) for the masked pixels
  p = zeros(M, K);

  % Initialize g(i,k) for the masked pixels
  g = zeros(M, K);

  % The nominator of Step 1
  nom_p = zeros(M, K);


% ------------------------------- Iterate L times ------------------------------
  for l = 1:L

% --------- Expectation: Compute probabilities P_ik using masked pixels --------
    for kernel = 1:K

      g(:,kernel) = g_k(image_masked_vec, centers(kernel,:), cov{kernel});

      nom_p(:,kernel) = w(kernel) * g(:, kernel);

    end


    % The denominator of Step 1
    sum_denom_p = sum(nom_p,2);


    for kernel = 1:K
      p(:,kernel) = nom_p(:,kernel) ./ sum_denom_p;
    end


% --- Maximization: Update weights, means and covariances using masked pixels --
    for kernel = 1:K
      w(kernel) = sum(p(:, kernel), 1) / M;

      centers(kernel, :) = p(:,kernel)' * image_masked_vec / (sum(p(:, kernel), 1));
    end

    for kernel = 1:K

      diff = bsxfun(@minus, image_masked_vec, centers(kernel, :));

      d = diff' * (diff .* repmat(p(:,kernel), [1 3]));

      diag_d = diag(diag(d));

      cov{kernel} =  (d / sum(p(:, kernel), 1)) * eye(3) + 0.0001 * diag(diag(rand(3)));
    end

  end


% ---------- Compute probabilities p(c_i) in Eq.(3) for all pixels I -----------
  pos_NK = zeros(N,K);

  for kernel = 1:K
    pos_NK(:,kernel) = g_k((image_vec), centers(kernel,:), cov{kernel}) * w(kernel);
    pos_NK(:,kernel) = pos_NK(:,kernel) / sum(pos_NK(:,kernel), 1);
  end

  pos = sum(pos_NK,2);

  prob = reshape(pos, height, width, 1);

end

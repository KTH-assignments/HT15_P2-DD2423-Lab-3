function g_k = g(pixel, center, sigma)

  diff = bsxfun(@minus, pixel, center);

  g_k = ((2 * pi)^3 * abs(det(sigma)))^(-1/2) * ...
    exp(-0.5 * sum(diff * inv(sigma) .* diff, 2));

end

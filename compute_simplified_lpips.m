function score = compute_simplified_lpips(img1, img2)
    % Convert images to grayscale (if needed)
    if size(img1, 3) == 3
        img1 = rgb2gray(img1);
    end
    if size(img2, 3) == 3
        img2 = rgb2gray(img2);
    end

    % Extract basic features using local standard deviation
    features1 = stdfilt(img1);
    features2 = stdfilt(img2);

    % Compute feature difference map
    feature_diff = abs(features1 - features2);

    % Compute the mean of the feature difference map as the LPIPS score
    score = mean(feature_diff(:));
end
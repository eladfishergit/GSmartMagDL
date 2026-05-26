% Assumes:
% x  -> Your true variable vector (e.g., Time, Energy, Distance)
% y1 -> ECDF of Group 1 evaluated at x
% y2 -> ECDF of Group 2 evaluated at x

function metrics = calculateMetrics(X, y1, y2)
    % Ensure they are column vectors
    x = X(:); y1 = y1(:); y2 = y2(:);
    dx = diff(x); % The steps of your true variable

    % 1. KS: Max vertical gap (Dimensionless)
    metrics.KS = max(abs(y1 - y2));

    % 2. Wasserstein: Area between curves (Units of X)% Assumes:
% x  -> Your true variable vector (e.g., Time, Energy, Distance)
% y1 -> ECDF of Group 1 evaluated at x
% y2 -> ECDF of Group 2 evaluated at x

function metrics = calculateMetrics(X, y1, y2)
    % Ensure they are column vectors
    x = x(:); y1 = y1(:); y2 = y2(:);
    dx = diff(x); % The steps of your true variable

    % 1. KS: Max vertical gap (Dimensionless)
    metrics.KS = max(abs(y1 - y2));

    % 2. Wasserstein: Area between curves (Units of X)
    % Integral of |y1 - y2| dx
    metrics.Wasserstein = sum(abs(y1(1:end-1) - y2(1:end-1)) .* dx);

    % 3. Cramer-von Mises: Integrated Square (Units of X)
    % Integral of (y1 - y2)^2 dx
    metrics.CvM = sum((y1(1:end-1) - y2(1:end-1)).^2 .* dx);

    % 4. Anderson-Darling (Approximate for ECDFs)
    % Weights the differences by the variance (1/F(1-F))
    avg_y = (y1 + y2) / 2;
    weight = 1 ./ (avg_y .* (1 - avg_y));
    weight(isinf(weight)) = 0; 
    
    metrics.AndersonDarling = sum(weight(1:end-1) .* (y1(1:end-1) - y2(1:end-1)).^2 .* dx);
end

    % Integral of |y1 - y2| dx
    metrics.Wasserstein = sum(abs(y1(1:end-1) - y2(1:end-1)) .* dx);

    % 3. Cramer-von Mises: Integrated Square (Units of X)
    % Integral of (y1 - y2)^2 dx
    metrics.CvM = sum((y1(1:end-1) - y2(1:end-1)).^2 .* dx);

    % 4. Anderson-Darling (Approximate for ECDFs)
    % Weights the differences by the variance (1/F(1-F))
    avg_y = (y1 + y2) / 2;
    weight = 1 ./ (avg_y .* (1 - avg_y));
    weight(isinf(weight)) = 0; 
    
    metrics.AndersonDarling = sum(weight(1:end-1) .* (y1(1:end-1) - y2(1:end-1)).^2 .* dx);
end

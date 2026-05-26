function [B,nm_traj,nm_reka] = buildMatrix(A, Q)
    % A: K x N x M array (with NaNs)
    % Q: scalar value for case when all NaN
    % B: N x M result

    [~, N, M] = size(A);
    B = ones(N, M)*Q;  % Preallocate
    k_traj = 0;
    k_reka = 0;
    
    for i = 1:N
        for j = 1:M
            vals = squeeze(A(:, i, j));         % All values across k
            validIdx = find(~isnan(vals)) ;   % Indices of non-NaN values
            
            if isempty(validIdx)
                % Case 2: all NaN
                B(i, j) = Q;
                k_reka = k_reka+1;
                nm_reka{k_reka} = [i,j];
            elseif numel(validIdx) == 1
                % Case 1: exactly one real value
                B(i, j) = vals(validIdx);
                k_traj = k_traj+1;
                nm_traj{k_traj} =[i,j] ;
            else
                % Case 3: more than one real value          
                k_traj = k_traj+1;
                nm_traj{k_traj} =[i,j] ;
                r = numel(validIdx);
                [~, maxIdx] = max(rand(1, r)) ; % Uniform random tie-break
                B(i, j) = vals(validIdx(maxIdx))  ;     
            end
        end
    end
end

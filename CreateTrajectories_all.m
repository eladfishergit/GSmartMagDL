close all
clear

amin  = -15;
amax = 15;

Ncurves = 1000;
Npts = 500;
dmin = 30;

for k = 1:Ncurves

    d = 0;
    while d < dmin
       
        P1 = amin+rand(1,2)*(amax-amin);
        P4 = [-P1(1), -P1(2)];  % Random point, you can adjust range
        d = norm(P1-P4);
    end
    
    % Given 3 points (you can modify these or use random ones)
    P2 = amin+rand(1,2)*(amax-amin); % Control point 1
    P3 = amin+rand(1,2)*(amax-amin);% Control point 2

    % Generate t values
    t = linspace(0, 1, Npts);

    % Calculate the cubic Bezier curve using the four control points
    x = (1 - t).^3 * P1(1) + 3 * (1 - t).^2 .* t * P2(1) + 3 * (1 - t) .* t.^2 * P3(1) + t.^3 * P4(1);
    y = (1 - t).^3 * P1(2) + 3 * (1 - t).^2 .* t * P2(2) + 3 * (1 - t) .* t.^2 * P3(2) + t.^3 * P4(2);

P(1:Npts,1) = x;
P(1:Npts,2) = y;
Traj{k}  = P;

end

save traj_data_all_30_10traj Traj


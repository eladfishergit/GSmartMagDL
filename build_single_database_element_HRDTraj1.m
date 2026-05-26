% This program generates a "theoretical" heatmap and the field sampled on a trajectory path.
% It then simulate the measurement of a smartphone magnetometer that walks
% on the path.

close all
clear
clc

% load traj_data1
load traj_data_all

Nmag_min = 1;
Nmag_max = 3;

kv = floor(rand*length(Traj));
P = Traj{kv};

% Earth magnetic field
Bxearth = 3e-6;
Byearth = 29e-6;
Bzearth = 33e-6;

% Densification of the traj.
Nd = 1; % Number of points to add between each pair
maxd = 250/1000; % max distance in mm for estimating the field on a grid from data on the path

% compute the Ground truth
% plot_flag = 1;
[Bxt,Byt,Bzt] = build_ground_truth1(Nmag_min,Nmag_max);
% load test_field

Bt = sqrt(Bxt.^2+Byt.^2+Bzt.^2);
Bt = Bt*1e6;
figure(1)
imagesc(Bt);
xlabel('X [samples]')
ylabel('Y [samples]')
set(gca,'YDir','normal')
title('Theoretical Calculation (in \muT)')
colorbar
set(gca,'YDir','normal')

Bxt_e = Bxt+Bxearth;
Byt_e = Byt+Byearth;
Bzt_e = Bzt+Bzearth;


% Sensitivity matrix
ds = rand(1,3)*1e-3;
S = diag(1-ds);

% pitch and roll variations - in degrees converted to radians
pitch_sigma = 2*pi/180;
roll_sigma = 2*pi/180;

% Grid
Xmax = 30;% meters
Ymax = 30;% meters
mgrid =300;

xg = linspace(-Xmax/2,Xmax/2,mgrid); % in m
yg = linspace(-Ymax/2,Ymax/2,mgrid); % in m

B_ground_truth = sqrt(Bxt_e.^2+Byt_e.^2+Bzt_e.^2)*1e6;
figure(2)
imagesc(xg,yg,B_ground_truth)
hold on
plot(P(:,1),P(:,2),'o')
xlabel('X [m]')
ylabel('Y [m]')
set(gca,'YDir','normal')
plot(P(:,1),P(:,2),'o')
title('Ground truth + Earth Mag Field  (in \muT)')
colorbar
colormap(jet)
axis equal;
xlim([-15 15]);
ylim([-15 15]);

xrot_o = P(:,1);
yrot_o = P(:,2);

Np = size(P,1);
% now sample the full (exact) field on the trajectory points
Bxtraj = zeros(1,Np);Bytraj = zeros(1,Np); Bztraj = zeros(1,Np);
Bxtraj_e = zeros(1,Np);Bytraj_e = zeros(1,Np); Bztraj_e = zeros(1,Np);

for i = 1:Np
    xi = xrot_o(i);
    yi = yrot_o(i);
    [~,ix] = min(abs(yi-xg));
    [~,ij] = min(abs(xi-yg));
    Bxtraj(i) = Bxt(ix,ij);
    Bytraj(i) = Byt(ix,ij);
    Bztraj(i) = Bzt(ix,ij);

    Bxtraj_e(i) = Bxt_e(ix,ij);
    Bytraj_e(i) = Byt_e(ix,ij);
    Bztraj_e(i) = Bzt_e(ix,ij);

end

Bttraj = sqrt(Bxtraj.^2+Bytraj.^2+Bztraj.^2);
Bttraj_e = sqrt(Bxtraj_e.^2+Bytraj_e.^2+Bztraj_e.^2);

figure(3)
plot(Bxtraj*1e6)
hold on
plot(Bytraj*1e6)
plot(Bztraj*1e6)
plot(Bttraj*1e6)
grid minor
xlabel('Time [samples]')
ylabel('Mag. Field [\muT]')
title('Ground truth sampled on the path')
legend('Bx','By','Bz','Btot')

% Pixelize the picture (e.g. squares of 1m x 1m)
windowSize = 2;   %  moving window size
xrot = xrot_o;
yrot = yrot_o;
Bxtraje = Bxtraj_e;
Bytraje = Bytraj_e;
Bztraje = Bztraj_e;
Bttraje = sqrt(Bxtraj_e.^2+Bytraj_e.^2+Bztraj_e.^2);

% Get dimensions of the input matrix
[rows, cols] = size(B_ground_truth);

% Calculate the size of the output matrix
outputRows = floor(rows / windowSize);
outputCols = floor(cols / windowSize);
rb =(0:outputRows-1)* windowSize + 1;
cb =(0:outputCols-1)* windowSize + 1;

output = NaN(outputRows,outputCols);
for i = 1:outputRows
    for j = 1:outputCols
        d = sqrt((xrot-xg(rb(j))).^2+(yrot-yg(cb(i))).^2);
        ind = find(d<=maxd);
        if ~isempty(ind)
            output(i,j) = mean(Bttraje(ind));
        end
    end
end

% Display the result
Bbgt = output;
Bbgt(isnan(output)) = median(output(~isnan(output)));

clear output

figure(4)
imagesc(xg,yg,Bbgt*1e6)
colormap(jet)
colorbar
set(gca,'YDir','normal')
xlabel('X [m]')
ylabel('Y [m]')
title('Map of Ground truth sampled on the path+earth MF')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% sort the pitch and roll angle within the given intervals
pitch = randn(1,Np)*pitch_sigma;
roll = randn(1,Np)*roll_sigma;

% Calibration matrix. It includes the misalignement and the soft-iron
% effect. The overall effect transform the "ideal" sphere defined by Bt
% when it is unperturbated by magnetic anomaly into an ellispsoid
dev_from_diag = 0.25;
dev_from_offdiag = 0.05;
rdiag = 1-dev_from_diag+2*rand(1,3)*dev_from_diag;
rodiag = -dev_from_offdiag+2*rand(1,6)*dev_from_offdiag;
T = diag(rdiag)+[0 rodiag(1:2);rodiag(3) 0 rodiag(4);rodiag(5:6) 0 ];


% read angles and prepare rotation matrix
dxr = diff(xrot);
dyr = diff(yrot);
% Exclude points with dxr = 0 since no angle can be defined for such values

ig = find(dxr~=0);
xroti  = xrot(ig);
yroti  = yrot(ig);
headg = atan2(diff(yroti),diff(xroti));
Npp = length(ig);
azi = zeros(1,Npp);
azi(1:end-1) = headg;
azi(end) =headg(end);

Bxtrajj = Bxtraj(ig) ;Bytrajj = Bytraj(ig) ;Bztrajj = Bztraj(ig) ;
Btraj = [Bxtrajj' Bytrajj' Bztrajj'];

% Gaussian noise epsilon
mu_noise = 0;
sigma_noise = 1; % microTesla
sigma_noise = sigma_noise*1e-6;
eps = mu_noise+randn(Npp,3)*sigma_noise;
Bxx = zeros(1,Npp);Byy = zeros(1,Npp);Bzz = zeros(1,Npp);

for k = 1:Npp
    ca  = cos(azi(k));
    sa = sin(azi(k));
    cb  = cos(pitch(k));
    sb = sin(pitch(k));
    cr  = cos(roll(k));
    sr = sin(roll(k));
    R = [ ca*cb  ca*sb*sr-sa*cr  ca*sb*cr+sa*sr; sa*cb sa*sb*sr+ca*cr sa*sb*cr-ca*sb; -sb cb*sr cb*cr];
    Bearth_rot = R*[Bxearth Byearth Bzearth]';
    bunrot(1) = Btraj(k,1);
    bunrot(2) = Btraj(k,2);
    bunrot(3) = Btraj(k,3);
    brot = R*bunrot';
    Ber(1) = brot(1)+Bearth_rot(1);
    Ber(2) = brot(2)+Bearth_rot(2);
    Ber(3) = brot(3)+Bearth_rot(3);

    bcal = S*(T*Ber');

    Bxx(k) = bcal(1) ;
    Byy(k) = bcal(2) ;
    Bzz(k) = bcal(3) ;

end

clear Bx By Bz

%add an offset (from hard iron and other source)
sigma_offset = 0.1; % in microTesla
sigma_offset = sigma_offset*1e-6;
Boffset = randn(1,3)*sigma_offset;

Bx = Bxx'+eps(:,1)+Boffset(1);
By = Byy'+eps(:,2)+Boffset(2);
Bz = Bzz'+eps(:,3)+Boffset(3);

Btt = sqrt(Bx.^2+By.^2+Bz.^2);

figure(5)
scatter(xroti, yroti,10,Btt*1e6,'filled')
colormap(jet)
colorbar
grid minor
xlabel('X [m]')
ylabel('Y [m]')
axis equal;
xlim([-15 15]);
ylim([-15 15]);
title('Mag. Field measured by phone on the path')

figure(6)
plot(Bx*1e6)
hold on
plot(By*1e6)
plot(Bz*1e6)
plot(Btt*1e6)
xlabel('Time [samples]')
ylabel('Mag. Field [\muT]')
title('Mag. Field measured by phone')
grid minor
legend('Bx','By','Bz','Btot')

% Pixelize the picture (e.g. squares of 1m x 1m)

% Get dimensions of the input matrix
[rows, cols] = size(B_ground_truth);

% Calculate the size of the output matrix
outputRows = floor(rows / windowSize);
outputCols = floor(cols / windowSize);
rb =(0:outputRows-1)* windowSize + 1;
cb =(0:outputCols-1)* windowSize + 1;

output = NaN(outputRows,outputCols);
for i = 1:outputRows
    for j = 1:outputCols
        d = sqrt((xroti-xg(rb(j))).^2+(yroti-yg(cb(i))).^2);
        ind = find(d<=maxd);
        if ~isempty(ind)
            output(i,j) = mean(Btt(ind));
        end
    end
end

% Display the result
Bb1 = output;
bad = isnan(output);
Bb1(bad==1) = median(B_ground_truth(:))/1e6;%median(output(~isnan(output)));
B_sim = Bb1*1e6;

figure(7)
imagesc(xg,yg,B_sim)
xlabel('X [m]')
ylabel('Y [m]')
colormap(jet)
colorbar
set(gca,'YDir','normal')
title('Map of Phone measurement (in \muT)')

min1 = min(B_ground_truth(:));
max1 = max(B_ground_truth(:));
min2 = min(B_sim(:));
max2 = max(B_sim(:));
cmin = min([min1 min2]);
cmax = max([max1 max2]);
figure(2)
clim([cmin cmax]);
figure(7)
clim([cmin cmax]);

figure(8)
imagesc(xg,yg,B_sim)
xlabel('X [m]')
ylabel('Y [m]')
colormap(jet)
colorbar
set(gca,'YDir','normal')
title('Map of Phone measurement (in \muT)')

% generate_report(ds, pitch_sigma, roll_sigma, T, Boffset)


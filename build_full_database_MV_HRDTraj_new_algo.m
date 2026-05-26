% This program generates a "theoretical" heatmap and the field sampled on a trajectory path.
% It then simulate the measurement of a smartphone magnetometer that walks
% on the path.

close all
clear
clc

% data_dir_root = 'T:\System\Elad-T\SmartMagDL-2\';
data_dir_root = 'P:\';

% load traj_data_all_30_10traj
load traj_data_all_30

Nmag_min = 3;
Nmag_max = 5;

% Earth magnetic field
% tel aviv
Bxearth = 3e-6;
Byearth = 29e-6;
Bzearth = 33e-6;

% Earth magnetic field
% Brazil north
% Bxearth = -4.67e-6;
% Byearth = 25.17e-6;
% Bzearth = 4.47e-6;
 
% % % Cape Town
% Bxearth = -4.7e-6;
% Byearth = 9.541e-6;
% Bzearth = -22.65e-6;


% % % Kursk
% Bxearth = 3.263e-6;
% Byearth = 18.668e-6;
% Bzearth = 48.405e-6;

% % India
% Bxearth = -1.244e-6;
% Byearth = 40.707e-6;
% Bzearth = 3.314e-6;

% % % % NewZealand
% Bxearth = 8.595e-6;
% Byearth = 17.438e-6;
% Bzearth = -55.457e-6;
% 
% % % Lapland
% Bxearth = 2.714e-6;
% Byearth =11.691e-6;
% Bzearth = 52.38e-6;

% % % NewOrleans
% Bxearth = -0.607e-6;
% Byearth = 24.02e-6;
% Bzearth = 39.597e-6;
% 
% % % Berlin
% Bxearth = 1.608e-6;
% Byearth = 18.608e-6;
% Bzearth = 46.33e-6;
% 
% % % Qaanaaq
% Bxearth = -2.755e-6;
% Byearth = 3.660e-6;
% Bzearth = 56.197e-6;

B_reka = sqrt(Bxearth^2+Byearth^2+Bzearth^2)*1e6;

maxd = 250/1000; % max distance in mm for estimating the field on a grid from data on the path
% pitch and roll variations - in degrees converted to radians
pitch_sigma = 2*pi/180;
roll_sigma = 2*pi/180;

%offset (from hard iron and other source)
sigma_offset = 0.1; % in microTesla
sigma_offset = sigma_offset*1e-6;

% Gaussian noise epsilon
mu_noise = 0;
sigma_noise = 1; % microTesla
sigma_noise = sigma_noise*1e-6;

% Calibration matrix. It includes the misalignement and the soft-iron
% effect. The overall effect transform the "ideal" sphere defined by Bt
% when it is unperturbated by magnetic anomaly into an ellispsoid
dev_from_diag = 0.25;
dev_from_offdiag = 0.05;

% Grid
Xmax = 30;% meters
Ymax = 30;% meters
mgrid =300;
xg = linspace(-Xmax/2,Xmax/2,mgrid); % in m
yg = linspace(-Ymax/2,Ymax/2,mgrid); % in m

% Pixelize the picture (e.g. squares of 1m x 1m)
windowSize = 1;   %  moving window size

% Start the main loop: over different maps (Nmaps groundtruths)
% then for each map Nviews will be created using Nviews trajectories.

Ntraining = 1000; % number for training
Nmaps = 100; % number for test
Nviews = 20; % # of different trajectories ove the same map

Ntotal = Ntraining*Nviews;
Nrequired = Nmaps*Nviews;

randomtraj_index = randperm(Ntotal,Nrequired);
m_run = 0;

f = waitbar(0, 'Starting');
for ksim = 1:Nmaps
    waitbar(ksim/Nmaps, f, sprintf('Progress: %d %%', floor(ksim/Nmaps*100)));

    % compute the Ground truth
    [Bxt,Byt,Bzt,Nobj] = build_ground_truth1(Nmag_min,Nmag_max);

    Bt = sqrt(Bxt.^2+Byt.^2+Bzt.^2);
    Bt = Bt*1e6;

    Bxt_e = Bxt+Bxearth;
    Byt_e = Byt+Byearth;
    Bzt_e = Bzt+Bzearth;

    Bxt_e_muT = Bxt_e*1e6;
    Byt_e_muT = Byt_e*1e6;
    Bzt_e_muT = Bzt_e*1e6;

    % Sensitivity matrix
    ds = rand(1,3)*1e-3;
    S = diag(1-ds);

    B_ground_truth = sqrt(Bxt_e.^2+Byt_e.^2+Bzt_e.^2)*1e6;
    B_theor = B_ground_truth;

    fname1 = ['Btheor_tot_',num2str(ksim)];
    out_name1 = [data_dir_root  'Labels_new_algo_test_telaviv\' fname1];
    out_name = [out_name1 '.mat'];

    save(out_name,'B_theor','Bxt_e_muT','Byt_e_muT','Bzt_e_muT','Nobj')

    B_all_views = zeros(Nviews,size(B_theor,1),size(B_theor,2));

    for kview = 1:Nviews

        % kv = (ksim-1)*Nviews+kview;
        m_run = m_run+1;
        kv = randomtraj_index(m_run);

        P = Traj{kv};

        xt = P(:,1);
        yt = P(:,2);

        Np = size(P,1);

        % now sample the full (exact) field on the trajectory points
        Bxtraj = zeros(1,Np);Bytraj = zeros(1,Np); Bztraj = zeros(1,Np);
        for i = 1:Np
            xi = xt(i);
            yi = yt(i);
            [~,ix] = min(abs(yi-xg));
            [~,ij] = min(abs(xi-yg));
            Bxtraj(i) = Bxt(ix,ij);
            Bytraj(i) = Byt(ix,ij);
            Bztraj(i) = Bzt(ix,ij);
        end

        % sort the pitch and roll angle within the given intervals
        pitch = randn(1,Np)*pitch_sigma;
        roll = randn(1,Np)*roll_sigma;

        % Calibration matrix
        rdiag = 1-dev_from_diag+2*rand(1,3)*dev_from_diag;
        rodiag = -dev_from_offdiag+2*rand(1,6)*dev_from_offdiag;
        T = diag(rdiag)+[0 rodiag(1:2);rodiag(3) 0 rodiag(4);rodiag(5:6) 0 ];

        % read angles and prepare rotation matrix
        dxr = diff(xt);
        dyr = diff(yt);

        % Exclude points with dxr = 0 since no angle can be defined for such values
        ig = find(dxr~=0);
        xroti  = xt(ig);
        yroti  = yt(ig);
        headg = atan2(diff(yroti),diff(xroti));
        Npp = length(ig);
        azi = zeros(1,Npp);
        azi(1:end-1) = headg;
        azi(end) = headg(end);

        Bxtrajj = Bxtraj(ig) ;Bytrajj = Bytraj(ig) ;Bztrajj = Bztraj(ig) ;
        Btraj = [Bxtrajj' Bytrajj' Bztrajj'];

        % Gaussian noise epsilon
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

        %add an offset (from hard iron and other source)
        Boffset = randn(1,3)*sigma_offset;
        Bx = Bxx'+eps(:,1)+Boffset(1);
        By = Byy'+eps(:,2)+Boffset(2);
        Bz = Bzz'+eps(:,3)+Boffset(3);

        Btt = sqrt(Bx.^2+By.^2+Bz.^2);

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

        % Replace the background (out off trajectory path) by a median of the field
        Bb1 = output;
        % bad = isnan(output);
        % Bb1(bad==1) = median(B_ground_truth(:))/1e6;
        B_sim_kv = Bb1*1e6;

        % fname2 = ['Bsim_tot_',num2str(ksim),'_',num2str(kview)];
        % out_name2= [data_dir_root  'Low_res\' fname2];
        % % out_name2= [data_dir_root  'low_res_test\' fname2];
        %
        % out_name = [out_name2 '.mat'];
        % save(out_name,'B_sim','azi')
        B_all_views(kview,:,:) = B_sim_kv;

    end

    [B_sim,NMtraj,NMreka] = buildMatrix(B_all_views, B_reka);

    fname2 = ['Bsim_tot_',num2str(ksim)];
    out_name2= [data_dir_root  'Low_res_new_algo_test_telaviv\' fname2];
    % out_name2= [data_dir_root  'low_res_test\' fname2];

    out_name = [out_name2 '.mat'];
    save(out_name,'B_sim','NMtraj','NMreka')

end

close(f)

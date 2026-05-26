
close all
clear
clc

load('MagneticDiverse50_2026.mat')

T = MagTable50;
Nmag_min = 3; 
Nmag_max = 5;

    % compute the Ground truth
    [Bxt,Byt,Bzt,Nobj] = build_ground_truth1(Nmag_min,Nmag_max);

 k = 1
    %
    % % Earth magnetic field
    Bxearth = Bx(k)/1e9;
    Byearth = By(k)/1e9;
    Bzearth = Bz(k)/1e9;

    Bt = sqrt(Bxt.^2+Byt.^2+Bzt.^2);
    Bt = Bt*1e6;

    Bxt_e = Bxt+Bxearth;
    Byt_e = Byt+Byearth;
    Bzt_e = Bzt+Bzearth;

    Bxt_e_muT = Bxt_e*1e6;
    Byt_e_muT = Byt_e*1e6;
    Bzt_e_muT = Bzt_e*1e6;

    B_ground_truth = sqrt(Bxt_e.^2+Byt_e.^2+Bzt_e.^2)*1e6;
    figure(k)
    imagesc(B_ground_truth)
    colorbar

pause

for k = 2:50
    %
    % % Earth magnetic field
    Bxearth = Bx(k)/1e9;
    Byearth = By(k)/1e9;
    Bzearth = Bz(k)/1e9;

    Bt = sqrt(Bxt.^2+Byt.^2+Bzt.^2);
    Bt = Bt*1e6;

    Bxt_e = Bxt+Bxearth;
    Byt_e = Byt+Byearth;
    Bzt_e = Bzt+Bzearth;

    Bxt_e_muT = Bxt_e*1e6;
    Byt_e_muT = Byt_e*1e6;
    Bzt_e_muT = Bzt_e*1e6;

    B_ground_truth = sqrt(Bxt_e.^2+Byt_e.^2+Bzt_e.^2)*1e6;
    figure(k)
    imagesc(B_ground_truth)
    colorbar
 
end
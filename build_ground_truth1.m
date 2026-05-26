function [bx,by,bz,Nobj] = build_ground_truth1(nmin,nmax)
%% Field computation


xmin = -15*1000;
xmax = 15*1000;
ymin = -15*1000;
ymax = 15*1000;

% Poses of the magnets: the first three coordinates (per line) are the
% positions, while the last three coordinates are the dipole magnetic
% moment unit vectors, e.g. the orientation of each magnet (in this case
% they are pointing parallel to xy plane of the "sensors grid")
% Nobj = 1; % pick 2 or 3 magnets
% Nobj = round(2*rand)+5; % pick 2 or 3 magnets
% Nobj = round(nmin + (nmax-nmin)*rand * nmin);% pick between nmin and nmax magnets
Nobj = round(nmin + (nmax-nmin)*rand);% pick between nmin and nmax magnets

dx = 5000; % inside the border (in millimeters)
dy = 5000; % inside the border (in millimeters)
xmmin = xmin+dx;xmmax = xmax-dx;
ymmin = ymin+dy;ymmax = ymax-dy;

or_ind = ceil(4*rand(1,Nobj));
orie =[0 1; 0 -1; 1 0; -1 0]; % 4 possible orientations of the magnets
% pick the first magnet Position and Orientation
x0 = xmmin+(xmmax-xmmin)*rand;
y0 = ymmin+(ymmax-ymmin)*rand;
z0 = -1000; % z offset in millimeter
P0 = [ x0 y0 z0];
M0 = [orie(or_ind(1),:) 0];
PM = [P0 M0];
MagPos = PM;
% pick the other magnets (1 or 2) Position and Orientation
for ko = 2:Nobj
    x0 = xmmin+(xmmax-xmmin)*rand;
    y0 = ymmin+(ymmax-ymmin)*rand;
    P0 = [ x0 y0 z0];
    M0 = [orie(or_ind(ko),:) 0];
    PM = [P0 M0];
    MagPos = [MagPos ; PM];
end

MagPos=MagPos';
MM = size(MagPos,2);

D = 25+5*rand(MM,1); % Diameter of the magnets
L = 800+400*rand(MM,1); % Length of the magnets

M = 1.2706/(4*pi*1e-7)*ones(MM,1); % Magnetization of the magnets [A/m]
fact = 1+rand(1,MM);
M = M./fact; % introduce a factor between 1 and 2 to get a proper field value (between 15 to 30 microTesla)

Ns = 300;
% Build a workspace of point of interest in planar arrangement below the
% plane of the magnets with Ns sensors along x and Ns sensors along y.
% Sensors lies in a equally spaced grid with step 0.1 m (100 mm).
[SensorPosMatrix,~] = buildWorkspace([Ns Ns],100,0);


% disp('Contemporary computation of the magnetic field in NsxNs points with ParallelGenerateReadings.m :')
% tic
Readings2 = ParallelGenerateReadings(D(1),L(1),M(1),MagPos,SensorPosMatrix');
Readings = Readings2';
% toc
B = reshape(Readings,[Ns,Ns,3]);
bx = B(:,:,1);
by = B(:,:,2);
bz = B(:,:,3);

% if flag_plot
%     figure(100)
%     Bt = sqrt(bx.^2+by.^2+bz.^2);
%     Bt = Bt*1e6;
%     figure(1)
%     imagesc(Bt);
%     xlabel('X [samples]')
%     ylabel('Y [samples]')
%     set(gca,'YDir','normal')
%     title('Theoretical Calculation (in \muT)')
%     colorbar
%     set(gca,'YDir','normal')
% end



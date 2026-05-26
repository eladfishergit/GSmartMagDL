clear
close all
clc

Bxearth = 3e-6;
Byearth = 29e-6;
Bzearth = 33e-6;
B_reka = sqrt(Bxearth^2+Byearth^2+Bzearth^2)*1e6;

work_dir = pwd;

% % % % %20 traj
% % % % % cd 'T:\System\Elad-T\SmartMagDL-2\Results\19102025_new_algo_test1\19102025'
% % % % % label_dir = 'Labels_new_algo_test1\';
% % % % % low_res_dir = 'Low_res_new_algo_test1\';
% % % % % load SRResNet_new_for_analysis
% % % %
% % % % % cd 'T:\System\Elad-T\SmartMagDL-2\Results\19102025_new_algo_test1\21102025\5'
% % % % % label_dir = 'Labels_new_algo_test_5traj\';
% % % % % low_res_dir = 'Low_res_new_algo_test_5traj\';
% % % % % load SRResNet_new_for_analysis
% % % %
% % % % % cd 'T:\System\Elad-T\SmartMagDL-2\Results\19102025_new_algo_test1\21102025\10'
% % % % % label_dir = 'Labels_new_algo_test_10traj\';
% % % % % low_res_dir = 'Low_res_new_algo_test_10traj\';
% % % % % load SRResNet_new_for_analysis
% % % %
% % % % % cd 'T:\System\Elad-T\SmartMagDL-2\Results\For Paper\SmartMagDL2_New_Run_batch_size_4_beta_0.05'
% % % % % cd 'T:\System\Elad-T\SmartMagDL-2\Results\For Paper\SmartMagDL2_New_Run_batch_size_4_beta_1';
% % % % % cd 'T:\System\Elad-T\SmartMagDL-2\Results\For Paper\SmartMagDL2_New_Run_batch_size_4_beta_0.05_ML_brazil'

% sitecase = 'Kursk';
% sitecase = 'Berlin';
% sitecase = 'India';
% sitecase = 'NewZealand';
% sitecase = 'NewOrleans';
% sitecase = 'Qaanaaq';
% sitecase = 'Lapland';
sitecase = 'telaviv';

data_run_test_dir = 'T:\System\Elad-T\SmartMagDL-2\Results\For Paper\';
flag = 1 ;    % SmartMagDl1
 flag = 2 ;   % SmartMagDl2 - new model KNN Entropy

if flag == 1
    dir_res = [data_run_test_dir,'SmartMagDL1_New_Run_batch_size_4_',sitecase];
else
    dir_res = [data_run_test_dir,'SmartMagDL2_New_Run_batch_size_4_beta_0.05_ML_kNN_entropy_',sitecase];
end
label_dir = ['Labels_new_algo_test_',sitecase,'\'];
low_res_dir = ['Low_res_new_algo_test_',sitecase,'\'];

cd(dir_res)

load SRResNet_new_for_analysis

cd(work_dir)


% cd 'T:\System\Elad-T\SmartMagDL-2\Results\For Paper\SmartMagDL2_New_Run_batch_size_4_beta_0.05_ML_kNN_entropy'
% cd 'T:\System\Elad-T\SmartMagDL-2\Results\For Paper\SmartMagDL2_New_Run_batch_size_4_beta_0.05_ML_kNN_entropy_brazil'
% cd 'T:\System\Elad-T\SmartMagDL-2\Results\For Paper\SmartMagDL2_New_Run_batch_size_4_beta_0.05_ML_kNN_entropy_capetown'

% cd 'T:\System\Elad-T\SmartMagDL-2\Results\For Paper\SmartMagDL1_New_Run_batch_size_4_CapeTown'
% cd 'T:\System\Elad-T\SmartMagDL-2\Results\For Paper\SmartMagDL1_New_Run_batch_size_4'
% cd 'T:\System\Elad-T\SmartMagDL-2\Results\For Paper\SmartMagDL1_New_Run_batch_size_4_brazil'

%
% label_dir = 'Labels_new_algo_test\';
% low_res_dir = 'Low_res_new_algo_test\';

% label_dir = 'Labels_new_algo_test_brazil\';
% low_res_dir = 'Low_res_new_algo_test_brazil\';
%
% label_dir = 'Labels_new_algo_test_CapeTown\';
% low_res_dir = 'Low_res_new_algo_test_CapeTown\';


data_dir_root = 'P:\';
Npts  = 300;
Nsim = 100;

B_theor_all =zeros(Nsim,Npts,Npts);
B_pred_all = zeros(Nsim,Npts,Npts);
% B_krig_all = zeros(Nsim,Npts,Npts);
B_sim_all = zeros(Nsim,Npts,Npts);

% Minterp = 600;
% B_cubic_all = zeros(Nsim,Minterp,Minterp);
% B_blin_all = zeros(Nsim,Minterp,Minterp);

f = waitbar(0, 'Starting');

for ksim = 1:Nsim

    waitbar(ksim/Nsim, f, sprintf('Progress: %d %%', floor(ksim/Nsim*100)));

    fname1 = ['Btheor_tot_',num2str(ksim)];
    out_name1 = [data_dir_root label_dir fname1];
    out_name = [out_name1 '.mat'];

    load(out_name,'B_theor')
    B_theor_all(ksim,:,:) = B_theor;

    B_pred_all(ksim,:,:) = double(squeeze(predictions(ksim,:,:)));

    fname2 = ['Bsim_tot_',num2str(ksim)];
    out_name2= [data_dir_root  low_res_dir fname2];
    out_name = [out_name2 '.mat'];
    load(out_name)
    B_sim_all(ksim,:,:) = B_sim;

    % ntraj = length(NMtraj);
    % mreka = length(NMreka);
    % % construct the set on which the Kriging interpolation will be performed
    % % first sample the field on the trajectories paths
    % for itraj = 1:ntraj
    %     q = NMtraj{itraj};
    %     x(itraj) = q(1);
    %     y(itraj) = q(2);
    %     z(itraj) = B_sim(q(1),q(2));
    % end
    %
    % nskip = 10;% take one each nskip from the trajectories
    % xx = x(1:nskip:end);
    % yy = y(1:nskip:end);
    % zz = z(1:nskip:end);
    %
    % ntraj = length(xx);
    % % first sample the field on the background to complete 50% of the whole map
    % % mreka is the total number of integers to choose from
    % nreka_sampled = 0.5*(Npts^2)-ntraj;  % Number of unique integers to pick
    %
    % % Use randperm to get the unique numbers
    % unique_nreka = randperm(mreka, nreka_sampled);
    % mm = 0;
    % nhop = 100; % take one each nhop from the background
    % for ireka = 1:nhop:nreka_sampled
    %     mm = mm+1;
    %     np = ntraj+mm;
    %     q = NMreka{unique_nreka(ireka)};
    %     xx(np) = q(1);
    %     yy(np) = q(2);
    %     zz(np) = B_reka;
    % end
    %
    % % calculate the sample variogram
    % v = variogram([xx' yy'],zz','plotit',false,'maxdist',100);
    % % and fit a spherical variogram
    % [~,~,~,vstruct] = variogramfit(v.distance,v.val,[],[],[],'model','stable','stablealpha',1);
    % % now use the sampled locations in a kriging
    % [X,Y] = meshgrid(1:Npts);
    % [Zhat,~] = kriging(vstruct,xx,yy,zz,X,Y);
    % bkr = Zhat';
    %
    %
    % B_krig_all(ksim,:,:) =  bkr;
    % [A_bilinear, A_cubic] = interpolateMatrix(B_sim, Minterp);
    % B_blin_all(ksim,:,:) = A_bilinear;
    % B_cubic_all(ksim,:,:) = A_cubic;
    % clearvars -except B_theor_all  B_pred_all B_krig_all B_sim_all B_cubic_all B_blin_all ksim Nsim Npts data_dir_root Minterp B_reka f predictions label_dir low_res_dir

    clearvars -except B_theor_all  B_pred_all  ksim Nsim B_sim_all Npts data_dir_root B_reka f predictions label_dir low_res_dir flag sitecase
end

% save results_test_20traj_MAGDL2_KNN_capetown B_theor_all  B_pred_all  B_sim_all

if flag == 1
    res_file = ['P:\Results_tests\results_test_20traj_MAGDL1_',sitecase];
else
    res_file = ['P:\Results_tests\results_test_20traj_MAGDL2_KNN_',sitecase];
end

save(res_file,'B_theor_all'  ,'B_pred_all'  ,'B_sim_all' )

% save results_test_20traj_MAGDL2_KNN_telaviv B_theor_all  B_pred_all  B_sim_all
% save results_test_20traj_MAGDL1_capetown B_theor_all  B_pred_all  B_sim_all
% save results_test_20traj_MAGDL1_telaviv B_theor_all  B_pred_all  B_sim_all

close(f)

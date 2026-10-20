import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.feature import peak_local_max
from skimage.measure import label, perimeter
from scipy.interpolate import griddata
from scipy.optimize import linear_sum_assignment
from pykrige.ok import OrdinaryKriging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.features = nn.Sequential(*list(vgg[:18])).eval()
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        return F.mse_loss(self.features(input), self.features(target))


def my_norm(arr):
    min_val, max_val = arr.min(), arr.max()
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def get_perceptual_tensor(img, device):
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img_tensor.repeat(1, 3, 1, 1).to(device)


def interpolate_sparse_map(sparse_map):
    vals, counts = np.unique(sparse_map[~np.isnan(sparse_map)], return_counts=True)
    bg_val = vals[np.argmax(counts)]
    mask = (sparse_map != bg_val) & (~np.isnan(sparse_map))
    if not np.any(mask):
        return sparse_map
    y, x = np.mgrid[0:sparse_map.shape[0], 0:sparse_map.shape[1]]
    points = np.column_stack((x[mask], y[mask]))
    interp = griddata(points, sparse_map[mask], (x, y), method='cubic')
    if np.any(np.isnan(interp)):
        interp_nearest = griddata(points, sparse_map[mask], (x, y), method='nearest')
        interp[np.isnan(interp)] = interp_nearest[np.isnan(interp)]
    return interp


def kriging_sparse_map(sparse_map, max_points=1500):
    vals, counts = np.unique(sparse_map[~np.isnan(sparse_map)], return_counts=True)
    bg_val = vals[np.argmax(counts)]
    mask = (sparse_map != bg_val) & (~np.isnan(sparse_map))
    if not np.any(mask):
        return sparse_map
    y, x = np.mgrid[0:sparse_map.shape[0], 0:sparse_map.shape[1]]
    px, py, pz = x[mask].astype(float), y[mask].astype(float), sparse_map[mask].astype(float)
    if len(px) > max_points:
        idx = np.random.choice(len(px), max_points, replace=False)
        px, py, pz = px[idx], py[idx], pz[idx]
    ok = OrdinaryKriging(px, py, pz, variogram_model='exponential', verbose=False, enable_plotting=False)
    z_krig, _ = ok.execute('grid', np.arange(sparse_map.shape[1], dtype=float),
                           np.arange(sparse_map.shape[0], dtype=float))
    return z_krig.data


def extract_anomalies(map_2d, pixel_size=0.1, min_dist_px=15, threshold_rel=0.2):
    """Extract physical metrics, including boundaries, integrals, perimeter, and polarity."""
    bg = np.median(map_2d)
    diff = np.abs(map_2d - bg)

    coords_px = peak_local_max(diff, min_distance=min_dist_px, threshold_rel=threshold_rel)
    if len(coords_px) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []

    threshold_abs = threshold_rel * diff.max()
    global_mask = diff >= threshold_abs
    labeled_mask = label(global_mask)

    pixel_area = pixel_size ** 2
    coords_m, amps, areas_m2, integrals, energies, perims_m, polarities, masks = [], [], [], [], [], [], [], []

    for (r, c) in coords_px:
        lbl = labeled_mask[r, c]
        if lbl == 0:
            obj_mask = np.zeros_like(map_2d, dtype=bool)
            obj_mask[r, c] = True
        else:
            obj_mask = (labeled_mask == lbl)

        amp = map_2d[r, c]
        coords_m.append([r * pixel_size, c * pixel_size])
        amps.append(amp)
        masks.append(obj_mask)

        areas_m2.append(np.sum(obj_mask) * pixel_area)
        integrals.append(np.sum(map_2d[obj_mask]) * pixel_area)
        energies.append(np.sum(map_2d[obj_mask] ** 2) * pixel_area)
        perims_m.append(perimeter(obj_mask) * pixel_size)
        polarities.append(1 if amp >= bg else -1)

    return (np.array(coords_m), np.array(amps), np.array(areas_m2),
            np.array(integrals), np.array(energies), np.array(perims_m),
            np.array(polarities), masks)


def evaluate_anomalies(gt_data, pr_data, max_match_dist_m=3.0):
    """Match anomalies and calculate all physical boundary and field errors."""
    gt_coords, gt_amps, gt_areas, gt_ints, gt_energies, gt_perims, gt_pols, gt_masks = gt_data
    pr_coords, pr_amps, pr_areas, pr_ints, pr_energies, pr_perims, pr_pols, pr_masks = pr_data

    if len(gt_coords) == 0 or len(pr_coords) == 0:
        return [], [], [], [], [], [], [], [], 0, len(pr_coords), len(gt_coords)

    dist_mat = np.linalg.norm(gt_coords[:, None, :] - pr_coords[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(dist_mat)

    loc_errs, amp_errs, bndry_errs, area_errs, int_errs, egy_errs, perim_errs, pol_errs = [], [], [], [], [], [], [], []
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if dist_mat[r, c] <= max_match_dist_m:
            tp += 1
            loc_errs.append(dist_mat[r, c])
            amp_errs.append(np.abs(gt_amps[r] - pr_amps[c]))
            area_errs.append(np.abs(gt_areas[r] - pr_areas[c]))
            int_errs.append(np.abs(gt_ints[r] - pr_ints[c]))
            egy_errs.append(np.abs(gt_energies[r] - pr_energies[c]))
            perim_errs.append(np.abs(gt_perims[r] - pr_perims[c]))
            pol_errs.append(0 if gt_pols[r] == pr_pols[c] else 1)

            # Boundary Error (1 - IoU)
            intersection = np.sum(gt_masks[r] & pr_masks[c])
            union = np.sum(gt_masks[r] | pr_masks[c])
            iou = intersection / union if union > 0 else 0.0
            bndry_errs.append(1.0 - iou)

    fp = len(pr_coords) - tp
    fn = len(gt_coords) - tp
    return loc_errs, amp_errs, bndry_errs, area_errs, int_errs, egy_errs, perim_errs, pol_errs, tp, fp, fn


def plot_ecdf(ax, data, label, color):
    if len(data) == 0: return
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, label=label, color=color, where='post', linewidth=2)


# ================= Configuration =================
models_paths = {
    'SmartMagDL2_20': r'T:\System\Elad-T\SmartMagDL-2\Results\all_traj_research_SmartMagDL2',
    'SmartMagDL1': r'T:\System\Elad-T\SmartMagDL-2\Results\all_traj_research_SmartMagDL1',
    'UNet': r'T:\System\Elad-T\SmartMagDL-2\Results\all_traj_research_UNet',
}
output_plots_base = r'T:\System\Elad-T\SmartMagDL-2\Results\Combined_Model_Comparisons_all_physics'
traj_list = ['5', '10', '20', '30', '50']

map_plotcases = ['RMSE', 'SSIM', 'PSNR', 'LPIPS']
anomaly_plotcases = [
    'Location_Error_m', 'Amplitude_Error_uT', 'Boundary_Error_1_minus_IoU',
    'Area_Error_m2', 'Boundary_Length_Error_m', 'Integral_Error', 'Energy_Error', 'Polarity_Error'
]
pic_size = 300

all_methods = list(models_paths.keys()) + ['Kriging', 'Cubic Interp']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
loss_fn_vgg = PerceptualLoss().to(device)

# ================= Summary Storage =================
summary_data = {plotcase: {method: [] for method in all_methods} for plotcase in map_plotcases + anomaly_plotcases}
summary_f1 = {method: [] for method in all_methods}
summary_precision = {method: [] for method in all_methods}
summary_recall = {method: [] for method in all_methods}
processed_trajs = []

# ================= Main Execution =================
for traj_file in traj_list:
    print(f"\n{'=' * 50}\nProcessing dataset: {traj_file} trajectories\n{'=' * 50}")

    first_model_name = list(models_paths.keys())[0]
    first_file_path = os.path.join(models_paths[first_model_name], traj_file, 'SRResNet_new_for_analysis.mat')
    if not os.path.exists(first_file_path):
        continue

    mat_data = scipy.io.loadmat(first_file_path)
    Ytest_all = np.squeeze(mat_data['Ytest'])
    Xtest_all = np.squeeze(mat_data['Xtest'])
    Nsim = Ytest_all.shape[0]

    predictions_all = {}
    skip_traj = False
    for model_name, path in models_paths.items():
        model_file = os.path.join(path, traj_file, 'SRResNet_new_for_analysis.mat')
        if not os.path.exists(model_file):
            skip_traj = True
            break
        predictions_all[model_name] = np.squeeze(scipy.io.loadmat(model_file)['predictions'])

    if skip_traj: continue
    processed_trajs.append(int(traj_file))

    # Setup Metrics
    map_metrics = {plotcase: {method: np.zeros(Nsim) for method in all_methods} for plotcase in map_plotcases}
    anomaly_metrics = {plotcase: {method: [] for method in all_methods} for plotcase in anomaly_plotcases}
    detection_stats = {method: {'TP': 0, 'FP': 0, 'FN': 0} for method in all_methods}

    print(f"Calculating metrics for {len(all_methods)} methods...")
    for i in range(Nsim):
        if i % 10 == 0: print(f"  Processed {i}/{Nsim} samples...")

        B_theor = Ytest_all[i].reshape(pic_size, pic_size)
        B_sim = Xtest_all[i].reshape(pic_size, pic_size)

        maps_to_evaluate = {
            'Cubic Interp': interpolate_sparse_map(B_sim),
            'Kriging': kriging_sparse_map(B_sim)
        }
        for m in models_paths.keys():
            maps_to_evaluate[m] = predictions_all[m][i].reshape(pic_size, pic_size)

        gt_data = extract_anomalies(B_theor)
        Btheo_n = my_norm(B_theor)
        tensor_theo = get_perceptual_tensor(Btheo_n, device)

        for method_name, B_pred in maps_to_evaluate.items():
            # 1. Map-Level Metrics
            Bpred_n = my_norm(B_pred)
            tensor_pred = get_perceptual_tensor(Bpred_n, device)

            with torch.no_grad():
                map_metrics['RMSE'][method_name][i] = np.sqrt(np.mean((Btheo_n - Bpred_n) ** 2))
                map_metrics['SSIM'][method_name][i] = ssim(Btheo_n, Bpred_n, data_range=1.0)
                map_metrics['PSNR'][method_name][i] = psnr(Btheo_n, Bpred_n, data_range=1.0)
                map_metrics['LPIPS'][method_name][i] = loss_fn_vgg(tensor_pred, tensor_theo).item()

            # 2. Anomaly-Level Metrics
            pr_data = extract_anomalies(B_pred)
            loc_err, amp_err, bndry_err, area_err, int_err, egy_err, perim_err, pol_err, tp, fp, fn = evaluate_anomalies(
                gt_data, pr_data)

            anomaly_metrics['Location_Error_m'][method_name].extend(loc_err)
            anomaly_metrics['Amplitude_Error_uT'][method_name].extend(amp_err)
            anomaly_metrics['Boundary_Error_1_minus_IoU'][method_name].extend(bndry_err)
            anomaly_metrics['Area_Error_m2'][method_name].extend(area_err)
            anomaly_metrics['Integral_Error'][method_name].extend(int_err)
            anomaly_metrics['Energy_Error'][method_name].extend(egy_err)
            anomaly_metrics['Boundary_Length_Error_m'][method_name].extend(perim_err)
            anomaly_metrics['Polarity_Error'][method_name].extend(pol_err)

            detection_stats[method_name]['TP'] += tp
            detection_stats[method_name]['FP'] += fp
            detection_stats[method_name]['FN'] += fn

    print(f"\n--- Anomaly Detection Performance ({traj_file} Trajectories) ---")
    for method in all_methods:
        tp, fp, fn = detection_stats[method]['TP'], detection_stats[method]['FP'], detection_stats[method]['FN']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"{method:>15}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f} (TP:{tp} FP:{fp} FN:{fn})")

        summary_f1[method].append(f1)
        summary_precision[method].append(precision)
        summary_recall[method].append(recall)

        for p_map in map_plotcases:
            summary_data[p_map][method].append(np.median(map_metrics[p_map][method]))
        for p_anom in anomaly_plotcases:
            if len(anomaly_metrics[p_anom][method]) > 0:
                summary_data[p_anom][method].append(np.median(anomaly_metrics[p_anom][method]))
            else:
                summary_data[p_anom][method].append(np.nan)

    # ================= Save Data & Plotting per Trajectory =================
    plots_dir = os.path.join(output_plots_base, traj_file)
    os.makedirs(plots_dir, exist_ok=True)

    metrics_to_save = {}
    for plotcase in map_plotcases:
        for method in all_methods:
            metrics_to_save[f"{plotcase}_{method.replace(' ', '_').replace('-', '_')}"] = map_metrics[plotcase][method]
    for plotcase in anomaly_plotcases:
        for method in all_methods:
            metrics_to_save[f"{plotcase}_{method.replace(' ', '_').replace('-', '_')}"] = np.array(
                anomaly_metrics[plotcase][method])
    for method in all_methods:
        safe_m = method.replace(' ', '_').replace('-', '_')
        metrics_to_save[f"TP_{safe_m}"] = detection_stats[method]['TP']
        metrics_to_save[f"FP_{safe_m}"] = detection_stats[method]['FP']
        metrics_to_save[f"FN_{safe_m}"] = detection_stats[method]['FN']

    scipy.io.savemat(os.path.join(plots_dir, f'Evaluation_Metrics_Data_{traj_file}traj.mat'), metrics_to_save)

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(all_methods))]

    for plotcase in map_plotcases + anomaly_plotcases:
        data_dict = map_metrics[plotcase] if plotcase in map_plotcases else anomaly_metrics[plotcase]
        all_data = np.concatenate([data_dict[m] for m in all_methods if len(data_dict[m]) > 0])
        if len(all_data) == 0: continue

        bin_edges = np.linspace(all_data.min(), all_data.max(), 51)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')
        width = (bin_centers[1] - bin_centers[0]) * 0.8
        depth = 0.4

        for idx, method in enumerate(all_methods):
            data = data_dict[method]
            if len(data) == 0: continue
            hist_vals, _ = np.histogram(data, bins=bin_edges)
            ax.bar3d(x=bin_centers - width / 2, y=np.full_like(bin_centers, idx) - depth / 2,
                     z=np.zeros_like(hist_vals),
                     dx=width, dy=depth, dz=hist_vals, color=colors[idx], alpha=0.6)

        display_plotcase = "Perceptual Loss (VGG)" if plotcase == 'LPIPS' else plotcase.replace('_', ' ')
        ax.set_title(f'Histogram - {display_plotcase} - {traj_file} trajectories')
        ax.set_xlabel('Value')
        ax.set_ylabel('Model')
        ax.set_zlabel('Frequency')
        ax.set_yticks(np.arange(len(all_methods)))
        ax.set_yticklabels(all_methods)
        ax.view_init(elev=15, azim=-45)

        legend_patches = [mpatches.Patch(color=colors[i], label=all_methods[i], alpha=0.6) for i in
                          range(len(all_methods))]
        ax.legend(handles=legend_patches, loc='best')
        plt.savefig(os.path.join(plots_dir, f'{plotcase}_Histogram_{traj_file}traj.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

        fig_ecdf, ax_ecdf = plt.subplots(figsize=(10, 7))
        for idx, method in enumerate(all_methods):
            plot_ecdf(ax_ecdf, data_dict[method], method, colors[idx])

        ax_ecdf.set_title(f'ECDF - {display_plotcase} - {traj_file} trajectories')
        ax_ecdf.set_xlabel('Value')
        ax_ecdf.set_ylabel('Cumulative Probability')
        ax_ecdf.grid(True, linestyle='--', alpha=0.7)
        ax_ecdf.legend(loc='best')
        plt.savefig(os.path.join(plots_dir, f'{plotcase}_ECDF_{traj_file}traj.png'), bbox_inches='tight', dpi=300)
        plt.close(fig_ecdf)

# ================= Summary Plotting =================
if len(processed_trajs) > 0:
    print(f"\n{'=' * 50}\nGenerating Summary Performance Plots\n{'=' * 50}")
    summary_dir = os.path.join(output_plots_base, 'Summary_Plots')
    os.makedirs(summary_dir, exist_ok=True)

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(all_methods))]

    for plotcase in map_plotcases + anomaly_plotcases:
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx, method in enumerate(all_methods):
            ax.plot(processed_trajs, summary_data[plotcase][method], marker='o', linewidth=2, label=method,
                    color=colors[idx])

        display_plotcase = "Perceptual Loss (VGG)" if plotcase == 'LPIPS' else plotcase.replace('_', ' ')
        ax.set_title(f'Median {display_plotcase} vs. Number of Trajectories')
        ax.set_xlabel('Number of Trajectories')
        ax.set_ylabel(f'Median {display_plotcase}')
        ax.set_xticks(processed_trajs)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best')
        plt.savefig(os.path.join(summary_dir, f'Summary_Median_{plotcase}.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, method in enumerate(all_methods):
        ax.plot(processed_trajs, summary_f1[method], marker='s', linewidth=2, label=method, color=colors[idx])
    ax.set_title('Anomaly Detection F1-Score vs. Number of Trajectories')
    ax.set_xlabel('Number of Trajectories')
    ax.set_ylabel('F1-Score')
    ax.set_xticks(processed_trajs)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    plt.savefig(os.path.join(summary_dir, 'Summary_F1_Score.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)

    print(f"Summary plots successfully saved to: {summary_dir}")
print("\nExecution Complete.")
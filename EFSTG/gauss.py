import os
import time
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib
import random
import gc
import util
from gaussmodel import EFSTG
matplotlib.use('Agg')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_adj_matrix(filepath, num_nodes, device):
    try:
        adj = np.load(filepath)
        if isinstance(adj, np.lib.npyio.NpzFile):
            adj = adj['adj'] if 'adj' in adj else adj[adj.files[0]]
        if adj.max() > 10:
            distances = adj[~np.isinf(adj)].flatten()
            std = distances.std()
            adj = np.exp(-np.square(adj / std))
            adj[adj < 1e-4] = 0.0
        adj = adj + np.eye(num_nodes)
        d = np.array(adj.sum(1))
        d_inv_sqrt = np.power(d, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return torch.tensor(normalized_adj, dtype=torch.float32).to(device)
    except Exception as e:
        print(f"邻接矩阵加载失败: {e}")
        return None

def add_gauss_noise(x, noise_level):
    if noise_level == 0:
        return x
    sigma = x.std(dim=(0, 2, 3), keepdim=True)
    noise = torch.randn_like(x) * (sigma * noise_level)
    return x + noise

TASKS = [
    {
        'DATA_NAME': 'PEMS08',
        'LR': 0.001,
        'WEIGHT_DECAY': 0.0001,
        'DROPOUT': 0.2,
        'CHANNELS': 64,
        'NOISE_LEVELS': [0.0, 0.2, 0.4, 0.6]
    },
{
        'DATA_NAME': 'PEMS03',
        'LR': 0.0018237448310985205,
        'WEIGHT_DECAY': 2.3467701398290323e-05,
        'DROPOUT': 0.2,
        'CHANNELS': 80,
        'NOISE_LEVELS': [0.0, 0.2, 0.4, 0.6]
    },
    {
        'DATA_NAME': 'PEMS04',
        'LR': 0.001996339887491085,
        'WEIGHT_DECAY': 4.829631649796517e-05,
        'DROPOUT': 0.3,
        'CHANNELS': 96,
        'NOISE_LEVELS': [0.0, 0.2, 0.4, 0.6]
    }
]
GLOBAL_SEED = 3407
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_EPOCHS = 15
PATIENCE = 5
ONECYCLE_EPOCHS = 15

if __name__ == "__main__":
    efficiency_results = []

    for task in TASKS:
        DATA_NAME = task['DATA_NAME']
        print(f"\n\n{'#' * 40}\n🚀 开始运行高斯噪声鲁棒性实验 - 数据集: {DATA_NAME}\n{'#' * 40}")

        start_pre = time.time()
        data_dict = util.load_dataset(f"data/processed/{DATA_NAME}", 32, 32, 32)
        data_dict['num_nodes'] = data_dict['x_train'].shape[2]
        scaler = data_dict['scaler']
        REAL_ADJ_TENSOR = load_adj_matrix(f'data/processed/{DATA_NAME}/{DATA_NAME.lower()}_distance_matrix.npy',
                                          data_dict['num_nodes'], DEVICE)
        end_pre = time.time()
        pre_processing_time = end_pre - start_pre

        RESULT_DIR = f"noise_study_{DATA_NAME}"
        os.makedirs(RESULT_DIR, exist_ok=True)
        all_noise_results = []

        for noise_rate in task['NOISE_LEVELS']:
            print(f"\n🚩 数据集: {DATA_NAME} | 当前高斯噪声等级: {int(noise_rate * 100)}%")
            set_seed(GLOBAL_SEED)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            model = EFSTG(
                device=DEVICE, input_dim=3, num_nodes=data_dict['num_nodes'],
                channels=task['CHANNELS'], granularity=288, dropout=task['DROPOUT'],
                real_adj=REAL_ADJ_TENSOR, ablation_mode='full'
            ).to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=task['LR'], weight_decay=task['WEIGHT_DECAY'])
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=task['LR'], epochs=ONECYCLE_EPOCHS,
                steps_per_epoch=len(data_dict['train_loader']), pct_start=0.2, anneal_strategy='cos'
            )

            epoch_times = []
            best_val_mae = float('inf')
            best_model_state = None
            patience_counter = 0

            for epoch in range(MAX_EPOCHS):
                model.train()
                start_epoch = time.time()

                for bx, by in data_dict['train_loader'].loader:
                    bx = bx.to(DEVICE).transpose(1, 3)
                    by = by.to(DEVICE).transpose(1, 3)[:, 0, :, :]

                    bx_noisy = add_gauss_noise(bx, noise_rate)

                    optimizer.zero_grad()
                    output, _ = model(bx_noisy)
                    pred = scaler.inverse_transform(output.squeeze(-1).permute(0, 2, 1))
                    real = scaler.inverse_transform(by)
                    loss = util.masked_mae_loss(pred, real, null_val=0.0)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    scheduler.step()

                end_epoch = time.time()
                epoch_times.append(end_epoch - start_epoch)

                model.eval()
                val_maes = []
                with torch.no_grad():
                    for bx, by in data_dict['val_loader'].loader:
                        bx = bx.to(DEVICE).transpose(1, 3)
                        by = by.to(DEVICE).transpose(1, 3)[:, 0, :, :]
                        output, _ = model(bx)
                        pred = scaler.inverse_transform(output.squeeze(-1).permute(0, 2, 1))
                        real = scaler.inverse_transform(by)
                        val_maes.append(util.MAE_torch(pred, real, 0.0).item())

                curr_val_mae = np.mean(val_maes)
                if curr_val_mae < best_val_mae:
                    best_val_mae = curr_val_mae
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                print(f"Epoch {epoch + 1:02d} | 时间: {epoch_times[-1]:.2f}s | 验证集 MAE: {curr_val_mae:.4f}")
                if patience_counter >= PATIENCE:
                    print("触发 Early Stopping!")
                    break

            model.load_state_dict(best_model_state)
            model.eval()

            start_inference = time.time()
            preds, trues = [], []
            with torch.no_grad():
                for bx, by in data_dict['test_loader'].loader:
                    bx = bx.to(DEVICE).transpose(1, 3)
                    by = by.to(DEVICE).transpose(1, 3)[:, 0, :, :]

                    bx_test_noisy = add_gauss_noise(bx, noise_rate)
                    output, _ = model(bx_test_noisy)

                    preds.append(scaler.inverse_transform(output.squeeze(-1).permute(0, 2, 1)).cpu())
                    trues.append(scaler.inverse_transform(by).cpu())

            end_inference = time.time()
            inference_time = end_inference - start_inference

            gpu_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3) if torch.cuda.is_available() else 0.0
            if noise_rate == 0.0:
                efficiency_results.append({
                    'Dataset': DATA_NAME,
                    'Training (avg/epoch)': f"{np.mean(epoch_times):.2f}s",
                    'Inference (total)': f"{inference_time:.2f}s",
                    'Pre-processing': f"{pre_processing_time:.2f}s",
                    'GPU Occupancy': f"{gpu_mem:.2f} GB"
                })
            all_preds = torch.cat(preds, dim=0)
            all_trues = torch.cat(trues, dim=0)
            avg_mae, avg_mape, avg_rmse, _ = util.metric(all_preds, all_trues)

            all_noise_results.append({
                'Noise_Level': f"{int(noise_rate * 100)}%",
                'MAE': round(float(avg_mae), 4),
                'RMSE': round(float(avg_rmse), 4),
                'MAPE(%)': round(float(avg_mape * 100), 4)
            })

            print(
                f"📊 噪声 {int(noise_rate * 100)}% 评价结果 -> MAE: {avg_mae:.4f} | RMSE: {avg_rmse:.4f} | MAPE: {avg_mape * 100:.4f}%")

            del model, optimizer, scheduler, best_model_state, all_preds, all_trues
            torch.cuda.empty_cache()
            gc.collect()

        df_noise = pd.DataFrame(all_noise_results)
        save_path = f"{RESULT_DIR}/noise_robustness_results.csv"
        df_noise.to_csv(save_path, index=False)

    print("\n" + "=" * 60)
    print(f"📈 PEMS08 高斯噪声鲁棒性实验结果已保存至: {save_path}")
    print("=" * 60)
    print(df_noise.to_markdown(index=False) if hasattr(df_noise, 'to_markdown') else df_noise)

    print("\n" + "=" * 60)
    print("📊 PEMS08 模型效率概览 (0% 噪声基线)")
    print("=" * 60)
    df_efficiency = pd.DataFrame(efficiency_results)
    print(df_efficiency.to_markdown(index=False) if hasattr(df_efficiency, 'to_markdown') else df_efficiency)
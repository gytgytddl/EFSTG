import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(42)
import time
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import util
from xrmodelx import TSGAT_Optimized
GLOBAL_SEED = 3407
def set_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_NAME = 'PEMS03'
MAX_EPOCHS = 20
PATIENCE = 5
LR = 0.0018237448310985205
WEIGHT_DECAY = 2.3467701398290323e-05
DROPOUT = 0.2
ONECYCLE_EPOCHS = 20
BATCH_SIZE = 32

BASE_CHANNELS = 80
BASE_LEVELS = 3
BASE_TOP_K = 15

TASK_LIST = [
    {'task_name': 'Full_Model', 'ablation_mode': 'full', 'channels': BASE_CHANNELS, 'tcn_levels': BASE_LEVELS,
     'top_k': BASE_TOP_K},

    {'task_name': 'Sens_Dim_32', 'ablation_mode': 'full', 'channels': 32, 'tcn_levels': BASE_LEVELS,
     'top_k': BASE_TOP_K},
    {'task_name': 'Sens_Dim_64', 'ablation_mode': 'full', 'channels': 64, 'tcn_levels': BASE_LEVELS,
     'top_k': BASE_TOP_K},
    {'task_name': 'Sens_Dim_128', 'ablation_mode': 'full', 'channels': 128, 'tcn_levels': BASE_LEVELS,
     'top_k': BASE_TOP_K},

    {'task_name': 'Sens_TCN_L2', 'ablation_mode': 'full', 'channels': BASE_CHANNELS, 'tcn_levels': 2,
     'top_k': BASE_TOP_K},
    {'task_name': 'Sens_TCN_L4', 'ablation_mode': 'full', 'channels': BASE_CHANNELS, 'tcn_levels': 4,
     'top_k': BASE_TOP_K},

    {'task_name': 'Sens_TopK_5', 'ablation_mode': 'full', 'channels': BASE_CHANNELS, 'tcn_levels': BASE_LEVELS,
     'top_k': 5},
    {'task_name': 'Sens_TopK_10', 'ablation_mode': 'full', 'channels': BASE_CHANNELS, 'tcn_levels': BASE_LEVELS,
     'top_k': 10},
    {'task_name': 'Sens_TopK_20', 'ablation_mode': 'full', 'channels': BASE_CHANNELS, 'tcn_levels': BASE_LEVELS,
     'top_k': 20},
]

if __name__ == "__main__":

    RESULT_DIR = f"xrresultsx/{DATA_NAME}"
    os.makedirs(RESULT_DIR, exist_ok=True)
    CSV_FILE = os.path.join(RESULT_DIR, f"Experiment_Results_{DATA_NAME}.csv")

    completed_tasks = []
    if os.path.exists(CSV_FILE):
        df_exist = pd.read_csv(CSV_FILE)
        if 'Task_Name' in df_exist.columns:
            completed_tasks = df_exist['Task_Name'].tolist()
            print(f"📦 发现历史进度，已完成 {len(completed_tasks)} 个任务。")

    print("=" * 50)
    print(f"🔥 开始加载 {DATA_NAME} 数据...")
    data_dict = util.load_dataset(f"data/processed/{DATA_NAME}", BATCH_SIZE, BATCH_SIZE, BATCH_SIZE)
    data_dict['num_nodes'] = data_dict['x_train'].shape[2]
    scaler = data_dict['scaler']

    if hasattr(scaler, 'mean') and hasattr(scaler, 'std'):
        np.save(f"{RESULT_DIR}/scaler_params.npy", {'mean': scaler.mean, 'std': scaler.std})

    def load_adj_matrix(filepath, num_nodes):
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
            return torch.tensor(normalized_adj, dtype=torch.float32).to(DEVICE)
        except:
            return None


    REAL_ADJ_TENSOR = load_adj_matrix(f'data/processed/{DATA_NAME}/{DATA_NAME.lower()}_distance_matrix.npy',
                                      data_dict['num_nodes'])

    for task in TASK_LIST:
        task_name = task['task_name']
        safe_name = task_name.replace('/', '_')

        if task_name in completed_tasks:
            print(f"⏭️ 任务 【{task_name}】 已存在于 CSV 中，自动跳过！")
            continue

        print("\n" + "🚀" * 15)
        print(f"🚀 开始训练任务: 【{task_name}】")
        print(
            f"🔧 参数配置: Mode={task['ablation_mode']}, Ch={task['channels']}, Lvl={task['tcn_levels']}, K={task['top_k']}")
        print("🚀" * 15)

        set_seed(GLOBAL_SEED)
        if hasattr(data_dict['train_loader'].loader, 'generator') and data_dict[
            'train_loader'].loader.generator is not None:
            data_dict['train_loader'].loader.generator.manual_seed(GLOBAL_SEED)

        model = TSGAT_Optimized(
            device=DEVICE,
            input_dim=3,
            num_nodes=data_dict['num_nodes'],
            channels=task['channels'],
            granularity=288,
            dropout=DROPOUT,
            real_adj=REAL_ADJ_TENSOR,
            ablation_mode=task['ablation_mode'],
            tcn_levels=task['tcn_levels'],
            top_k=task['top_k']
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        steps_per_epoch = len(data_dict['train_loader'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LR, epochs=ONECYCLE_EPOCHS, steps_per_epoch=steps_per_epoch,
            pct_start=0.2, anneal_strategy='cos'
        )

        best_val_mae = float('inf')
        best_model_state = None
        patience_counter = 0
        best_epoch = 0
        train_time_list = []
        history_train_loss = []
        history_val_mae = []

        torch.cuda.reset_peak_memory_stats()

        for epoch in range(MAX_EPOCHS):
            epoch_start = time.time()
            model.train()
            train_loss = []

            for bx, by in data_dict['train_loader'].loader:
                bx, by = bx.to(DEVICE).transpose(1, 3), by.to(DEVICE).transpose(1, 3)[:, 0, :, :]
                optimizer.zero_grad()
                output, _ = model(bx)

                pred = scaler.inverse_transform(output.squeeze(-1).permute(0, 2, 1))
                real = scaler.inverse_transform(by)

                loss = util.masked_mae_loss(pred, real, null_val=0.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                if epoch < ONECYCLE_EPOCHS:
                    scheduler.step()

                train_loss.append(loss.item())

            train_time_list.append(time.time() - epoch_start)

            model.eval()
            val_maes = []
            with torch.no_grad():
                for bx, by in data_dict['val_loader'].loader:
                    bx, by = bx.to(DEVICE).transpose(1, 3), by.to(DEVICE).transpose(1, 3)[:, 0, :, :]
                    output, _ = model(bx)
                    pred = scaler.inverse_transform(output.squeeze(-1).permute(0, 2, 1))
                    real = scaler.inverse_transform(by)
                    val_maes.append(util.MAE_torch(pred, real, 0.0).item())

            current_val_mae = np.mean(val_maes)
            current_train_loss = np.mean(train_loss)
            current_lr = optimizer.param_groups[0]['lr']
            history_train_loss.append(current_train_loss)
            history_val_mae.append(current_val_mae)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"[{task_name}] Epoch {epoch + 1:03d} | Train Loss: {current_train_loss:.4f} | Val MAE: {current_val_mae:.4f}")

            if current_val_mae < best_val_mae:
                best_val_mae = current_val_mae
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(best_model_state, f"{RESULT_DIR}/best_model_{safe_name}.pth")
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"🛑 触发早停! 最佳 Epoch 为 {best_epoch}, 最佳 Val MAE: {best_val_mae:.4f}")
                break

        gpu_max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print(f"📊 加载 {task_name} 的最佳权重进行最终打分...")
        model.load_state_dict(best_model_state)
        model.eval()

        preds, trues = [], []
        dynamic_adjs, static_adjs, fusion_gates = [], [], []

        inf_start_time = time.time()

        with torch.no_grad():
            for bx, by in data_dict['test_loader'].loader:
                bx, by = bx.to(DEVICE).transpose(1, 3), by.to(DEVICE).transpose(1, 3)[:, 0, :, :]
                output, info_dict = model(bx)
                preds.append(scaler.inverse_transform(output.squeeze(-1).permute(0, 2, 1)).cpu())
                trues.append(scaler.inverse_transform(by).cpu())

                if 'dynamic_adj' in info_dict:
                    dynamic_adjs.append(info_dict['dynamic_adj'][-1].cpu().numpy())
                if 'static_adj' in info_dict:
                    static_adjs.append(info_dict['static_adj'][-1].cpu().numpy())
                if 'fusion_gate' in info_dict:
                    fusion_gates.append(info_dict['fusion_gate'][-1].cpu().numpy())

        total_inf_time = time.time() - inf_start_time

        all_preds = torch.cat(preds, dim=0)
        all_trues = torch.cat(trues, dim=0)

        test_mae, test_mape, test_rmse, _ = util.metric(all_preds, all_trues)

        if task_name == 'Full_Model' or "w/o" in task_name:
            np.save(f"{RESULT_DIR}/predicted_{safe_name}.npy", all_preds.numpy())
            if len(dynamic_adjs) > 0:
                np.save(f"{RESULT_DIR}/dynamic_adj_{safe_name}.npy", np.array(dynamic_adjs))
            if len(static_adjs) > 0:
                np.save(f"{RESULT_DIR}/static_adj_{safe_name}.npy", np.array(static_adjs))
            if len(fusion_gates) > 0:
                np.save(f"{RESULT_DIR}/fusion_gate_{safe_name}.npy", np.array(fusion_gates))
            if task_name == 'Full_Model':
                np.save(f"{RESULT_DIR}/true_test.npy", all_trues.numpy())

        plt.figure(figsize=(10, 5))
        plt.plot(history_train_loss, label='Train Loss')
        plt.plot(history_val_mae, label='Val MAE')
        plt.title(f'Loss Curve - {task_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/MAE')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{RESULT_DIR}/loss_curve_{safe_name}.png", dpi=300)
        plt.close()

        horizon_metrics = {}
        for h_idx, step_name in zip([2, 5, 11], ["15min", "30min", "60min"]):
            h_mae = util.MAE_torch(all_preds[:, :, h_idx], all_trues[:, :, h_idx], 0.0).item()
            h_rmse = util.RMSE_torch(all_preds[:, :, h_idx], all_trues[:, :, h_idx], 0.0).item()
            h_mape = util.MAPE_torch(all_preds[:, :, h_idx], all_trues[:, :, h_idx], 0.0).item() * 100
            horizon_metrics.update({
                f'{step_name}_MAE': round(h_mae, 4),
                f'{step_name}_RMSE': round(h_rmse, 4),
                f'{step_name}_MAPE(%)': round(h_mape, 4)
            })

        print(f"🏆 【{task_name}】测试结果: MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f} | MAPE: {test_mape * 100:.4f}%")
        print(
            f"⏱️ 平均训练时间/Epoch: {np.mean(train_time_list):.2f}s | 总推理时间: {total_inf_time:.2f}s | 峰值显存: {gpu_max_mem:.2f} GB")


        record = {
            'Task_Name': task_name,
            'Best_Epoch': best_epoch,
            'Global_MAE': round(test_mae, 4),
            'Global_RMSE': round(test_rmse, 4),
            'Global_MAPE(%)': round(test_mape * 100, 4),
            **horizon_metrics,
            'Params(M)': round(model.param_num(), 4),
            'TrainTime/Epoch(s)': round(np.mean(train_time_list), 2),
            'Total_Inference(s)': round(total_inf_time, 2),
            'Max_GPU(GB)': round(gpu_max_mem, 2)
        }

        df_current = pd.DataFrame([record])
        write_header = not os.path.exists(CSV_FILE)
        df_current.to_csv(CSV_FILE, mode='a', header=write_header, index=False)
        print(f"💾 当前结果已实时追加到: {CSV_FILE}")

    print("\n" + "=" * 60)
    print("=" * 60)
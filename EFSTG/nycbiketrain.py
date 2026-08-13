import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONHASHSEED'] = str(42)

import random
import torch
import torch.nn.functional as F
import numpy as np
import copy
import optuna
import utiln
from nycbikemodel import EFSTG

GLOBAL_SEED = 3407


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(GLOBAL_SEED)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_NAME = 'NYC-BIKE'
EPOCHS = 15
PATIENCE = 5
DB_NAME = f"sqlite:///rNBcan_{DATA_NAME.replace('-', '_')}_v3.db"
STUDY_NAME = "rNBcan_Pure_Search_v3"

print("=" * 50)
print(f"开始加载 {DATA_NAME} 数据... 使用基准种子: {GLOBAL_SEED}")
data_dict = utiln.load_dataset(f"data/processed/{DATA_NAME}", 16, 16, 16)
data_dict['num_nodes'] = data_dict['x_train'].shape[2]
scaler = data_dict['scaler']
print("=" * 50)

REAL_ADJ_TENSOR = None


def objective(trial):
    set_seed(GLOBAL_SEED)

    if hasattr(data_dict['train_loader'].loader, 'generator') and data_dict[
        'train_loader'].loader.generator is not None:
        data_dict['train_loader'].loader.generator.manual_seed(GLOBAL_SEED)

    channels = trial.suggest_categorical("channels", [64, 96, 128])
    lr = trial.suggest_float("lr", 0.001, 0.0038, log=True)
    weight_decay = trial.suggest_float("weight_decay", 7e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    k = trial.suggest_categorical("k", [8, 10, 15])
    gc_gate_init = trial.suggest_categorical("gc_gate_init", [0.1, 0.2, 0.3])
    graph_alpha_init = trial.suggest_categorical("graph_alpha_init", [0.01, 0.1, 0.3])

    print(f"\n---> [Trial {trial.number}] 完整参数配置: "
          f"Channels={channels}, LR={lr:.5f}, Drop={dropout:.2f}, "
          f"WD={weight_decay:.5f}, k={k}, GC_init={gc_gate_init}, Alpha_init={graph_alpha_init}")
    model = EFSTG(
        device=DEVICE,
        input_dim=2,
        out_dim=2,
        num_nodes=data_dict['num_nodes'],
        channels=channels,
        granularity=48,
        dropout=dropout,
        real_adj=REAL_ADJ_TENSOR,
        k=k,
        gc_gate_init=gc_gate_init,
        graph_alpha_init=graph_alpha_init
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(data_dict['train_loader'])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
        pct_start=0.2, anneal_strategy='cos'
    )

    best_val_mae = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        try:
            for bx, by in data_dict['train_loader'].loader:
                bx = bx.to(DEVICE).transpose(1, 3)
                by = by.to(DEVICE).transpose(1, 3)[:, :2, :, :]

                optimizer.zero_grad()
                output, _ = model(bx)

                pred = scaler.inverse_transform(output.transpose(1, 3)).transpose(1, 3)
                real = scaler.inverse_transform(by.transpose(1, 3)).transpose(1, 3)

                loss = utiln.masked_mae_loss(pred, real, null_val=0.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                scheduler.step()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[Trial {trial.number}] OOM 显存溢出，跳过。")
                torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()
            else:
                raise e

        model.eval()
        val_maes, val_pk_maes, val_dp_maes = [], [], []
        with torch.no_grad():
            for bx, by in data_dict['val_loader'].loader:
                bx = bx.to(DEVICE).transpose(1, 3)
                by = by.to(DEVICE).transpose(1, 3)[:, :2, :, :]

                output, _ = model(bx)
                pred = scaler.inverse_transform(output.transpose(1, 3))
                real = scaler.inverse_transform(by.transpose(1, 3))

                val_maes.append(utiln.MAE_torch(pred, real, 0.0).item())
                val_pk_maes.append(utiln.MAE_torch(pred[..., 0], real[..., 0], 0.0).item())
                val_dp_maes.append(utiln.MAE_torch(pred[..., 1], real[..., 1], 0.0).item())
        current_val_mae = np.mean(val_maes)
        current_val_pk_mae = np.mean(val_pk_maes)
        current_val_dp_mae = np.mean(val_dp_maes)
        trial.report(current_val_mae, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if current_val_mae < best_val_mae:
            best_val_mae = current_val_mae
            best_val_pk_mae = current_val_pk_mae
            best_val_dp_mae = current_val_dp_mae
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"[Trial {trial.number}] 触发早停，停止在 Epoch {epoch + 1}")
            break

    model.load_state_dict(best_model_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in data_dict['test_loader'].loader:
            bx = bx.to(DEVICE).transpose(1, 3)
            by = by.to(DEVICE).transpose(1, 3)[:, :2, :, :]
            output, _ = model(bx)

            preds.append(scaler.inverse_transform(output.transpose(1, 3)).cpu())
            trues.append(scaler.inverse_transform(by.transpose(1, 3)).cpu())

    pred_cat = torch.cat(preds, dim=0)
    true_cat = torch.cat(trues, dim=0)

    pk_mae, pk_mape, pk_rmse, _ = utiln.metric(pred_cat[..., 0], true_cat[..., 0])
    dp_mae, dp_mape, dp_rmse, _ = utiln.metric(pred_cat[..., 1], true_cat[..., 1])

    all_mae, all_mape, all_rmse, _ = utiln.metric(pred_cat, true_cat)

    trial.set_user_attr("Val_Pick_MAE", best_val_pk_mae)
    trial.set_user_attr("Val_Drop_MAE", best_val_dp_mae)
    trial.set_user_attr("Val_All_MAE", best_val_mae)

    trial.set_user_attr("Test_Pick_MAE", pk_mae)
    trial.set_user_attr("Test_Pick_RMSE", pk_rmse)
    trial.set_user_attr("Test_Pick_MAPE", pk_mape)

    trial.set_user_attr("Test_Drop_MAE", dp_mae)
    trial.set_user_attr("Test_Drop_RMSE", dp_rmse)
    trial.set_user_attr("Test_Drop_MAPE", dp_mape)

    trial.set_user_attr("Test_All_MAE", all_mae)

    print(f"\n🔥 [Trial {trial.number} 任务完成] 测试集结果:")
    print(
        f"   🚕 [Pick-Up]  Val MAE: {best_val_pk_mae:.4f} | Test MAE: {pk_mae:.4f} | Test RMSE: {pk_rmse:.4f} | Test MAPE: {pk_mape * 100:.4f}%")
    print(
        f"   🚖 [Drop-Off] Val MAE: {best_val_dp_mae:.4f} | Test MAE: {dp_mae:.4f} | Test RMSE: {dp_rmse:.4f} | Test MAPE: {dp_mape * 100:.4f}%")
    print(
        f"   🌐 [Overall]  Val MAE: {best_val_mae:.4f} | Test MAE: {all_mae:.4f} | Test RMSE: {all_rmse:.4f} | Test MAPE: {all_mape * 100:.4f}%")
    return best_val_mae


if __name__ == "__main__":
    print("\n🚀 启动纯净版 SOTA 冲刺 (支持双输出分离评测) 🚀")

    sampler = optuna.samplers.TPESampler(seed=GLOBAL_SEED)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=DB_NAME,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10, interval_steps=2)
    )

    study.optimize(objective, n_trials=30, catch=(RuntimeError,))

    print("\n🏆 最终最佳结果 (按验证集指标选出) 🏆")
    best_trial = study.best_trial

    print("\n[最佳 Trial 使用的参数]:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    print("\n[最佳 Trial 的表现]:")
    print(
        f"   🚕 [Pick-Up]  Val MAE: {best_trial.user_attrs.get('Val_Pick_MAE', 'N/A'):.4f} | Test MAE: {best_trial.user_attrs.get('Test_Pick_MAE', 'N/A'):.4f}")
    print(
        f"   🚖 [Drop-Off] Val MAE: {best_trial.user_attrs.get('Val_Drop_MAE', 'N/A'):.4f} | Test MAE: {best_trial.user_attrs.get('Test_Drop_MAE', 'N/A'):.4f}")
    print(
        f"   🌐 [Overall]  Val MAE: {best_trial.value:.4f} | Test MAE: {best_trial.user_attrs.get('Test_All_MAE', 'N/A'):.4f}")
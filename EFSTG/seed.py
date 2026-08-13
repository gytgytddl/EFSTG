import os
import time
import random
import copy
import gc
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import numpy as np
import pandas as pd
import utiln
from seedm import EFSTG
SEEDS = [42, 100, 2023, 2024, 3407]
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
def to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DATA_NAME = 'NYC-BIKE'
EPOCHS = 15
PATIENCE = 5
INPUT_LEN = 12
OUTPUT_LEN = 12
BATCH_SIZE = 16
BEST_PARAMS = {
    'channels': 96,
    'lr': 0.0028297369824960964,
    'weight_decay': 3.919812543135198e-05,
    'dropout': 0.1336373118764793,
    'k': 15,
    'gc_gate_init': 0.2,
    'graph_alpha_init': 0.1,
    'tcn_levels': 3
}

def calculate_metrics(pred_cat, true_cat):
    all_mae, all_mape, all_rmse, _ = utiln.metric(pred_cat, true_cat)
    pk_mae, pk_mape, pk_rmse, _ = utiln.metric(
        pred_cat[..., 0],
        true_cat[..., 0]
    )
    dp_mae, dp_mape, dp_rmse, _ = utiln.metric(
        pred_cat[..., 1],
        true_cat[..., 1]
    )

    s3_all_mae, s3_all_mape, s3_all_rmse, _ = utiln.metric(
        pred_cat[:, 2, :, :],
        true_cat[:, 2, :, :]
    )
    s6_all_mae, s6_all_mape, s6_all_rmse, _ = utiln.metric(
        pred_cat[:, 5, :, :],
        true_cat[:, 5, :, :]
    )

    s12_all_mae, s12_all_mape, s12_all_rmse, _ = utiln.metric(
        pred_cat[:, 11, :, :],
        true_cat[:, 11, :, :]
    )

    s3_pk_mae, s3_pk_mape, s3_pk_rmse, _ = utiln.metric(
        pred_cat[:, 2, :, 0],
        true_cat[:, 2, :, 0]
    )

    s6_pk_mae, s6_pk_mape, s6_pk_rmse, _ = utiln.metric(
        pred_cat[:, 5, :, 0],
        true_cat[:, 5, :, 0]
    )

    s12_pk_mae, s12_pk_mape, s12_pk_rmse, _ = utiln.metric(
        pred_cat[:, 11, :, 0],
        true_cat[:, 11, :, 0]
    )

    s3_dp_mae, s3_dp_mape, s3_dp_rmse, _ = utiln.metric(
        pred_cat[:, 2, :, 1],
        true_cat[:, 2, :, 1]
    )

    s6_dp_mae, s6_dp_mape, s6_dp_rmse, _ = utiln.metric(
        pred_cat[:, 5, :, 1],
        true_cat[:, 5, :, 1]
    )

    s12_dp_mae, s12_dp_mape, s12_dp_rmse, _ = utiln.metric(
        pred_cat[:, 11, :, 1],
        true_cat[:, 11, :, 1]
    )

    metrics = {
        'All_MAE': to_float(all_mae),
        'All_RMSE': to_float(all_rmse),
        'All_MAPE(%)': to_float(all_mape) * 100,

        'Pick_MAE': to_float(pk_mae),
        'Pick_RMSE': to_float(pk_rmse),
        'Pick_MAPE(%)': to_float(pk_mape) * 100,

        'Drop_MAE': to_float(dp_mae),
        'Drop_RMSE': to_float(dp_rmse),
        'Drop_MAPE(%)': to_float(dp_mape) * 100,

        'S3_All_MAE': to_float(s3_all_mae),
        'S3_All_RMSE': to_float(s3_all_rmse),

        'S6_All_MAE': to_float(s6_all_mae),
        'S6_All_RMSE': to_float(s6_all_rmse),

        'S12_All_MAE': to_float(s12_all_mae),
        'S12_All_RMSE': to_float(s12_all_rmse),

        'S3_Pick_MAE': to_float(s3_pk_mae),
        'S6_Pick_MAE': to_float(s6_pk_mae),
        'S12_Pick_MAE': to_float(s12_pk_mae),

        'S3_Drop_MAE': to_float(s3_dp_mae),
        'S6_Drop_MAE': to_float(s6_dp_mae),
        'S12_Drop_MAE': to_float(s12_dp_mae),
    }
    return metrics

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print(f"🚀 启动 {DATA_NAME} 多随机种子实验")
    print("=" * 70)
    print(f"📌 Device: {DEVICE}")
    print(f"📌 Seeds: {SEEDS}")
    print(f"📌 Batch Size: {BATCH_SIZE}")
    print(f"📌 Epochs: {EPOCHS}")
    print(f"📌 Patience: {PATIENCE}")
    print("=" * 70)

    RESULT_DIR = f"MultiSeed/{DATA_NAME}"
    os.makedirs(RESULT_DIR, exist_ok=True)
    CSV_FILE = os.path.join(
        RESULT_DIR,
        f"Multi_Seed_Results_{DATA_NAME}.csv"
    )
    set_seed(SEEDS[0])
    print(f"\n⏳ 开始加载 {DATA_NAME} 数据集...")
    data_dict = utiln.load_dataset(
        f"data/processed/{DATA_NAME}",
        BATCH_SIZE,
        BATCH_SIZE,
        BATCH_SIZE
    )

    data_dict['num_nodes'] = data_dict['x_train'].shape[2]
    scaler = data_dict['scaler']

    print(f"✅ 数据加载完成")
    print(f"📌 节点数量: {data_dict['num_nodes']}")
    print(f"📌 训练批次数量: {len(data_dict['train_loader'])}")
    print(f"📌 验证批次数量: {len(data_dict['val_loader'])}")
    print(f"📌 测试批次数量: {len(data_dict['test_loader'])}")

    if hasattr(scaler, 'mean') and hasattr(scaler, 'std'):
        np.save(
            os.path.join(RESULT_DIR, "scaler_params.npy"),
            {
                'mean': scaler.mean,
                'std': scaler.std
            }
        )
    REAL_ADJ_TENSOR = None
    all_seed_records = []
    for run_idx, seed in enumerate(SEEDS, start=1):
        print("\n" + "🚀" * 25)
        print(f"🚀 运行进度: [{run_idx}/{len(SEEDS)}]")
        print(f"🚀 当前随机种子: {seed}")
        print("🚀" * 25)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        set_seed(seed)

        if (
            hasattr(data_dict['train_loader'].loader, 'generator')
            and data_dict['train_loader'].loader.generator is not None
        ):
            data_dict['train_loader'].loader.generator.manual_seed(seed)

        model = EFSTG(
            device=DEVICE,
            input_dim=2,
            out_dim=2,
            num_nodes=data_dict['num_nodes'],
            channels=BEST_PARAMS['channels'],
            input_len=INPUT_LEN,
            output_len=OUTPUT_LEN,
            granularity=48,
            dropout=BEST_PARAMS['dropout'],
            real_adj=REAL_ADJ_TENSOR,
            k=BEST_PARAMS['k'],
            tcn_levels=BEST_PARAMS['tcn_levels'],
            gc_gate_init=BEST_PARAMS['gc_gate_init'],
            graph_alpha_init=BEST_PARAMS['graph_alpha_init'],
            ablation_mode='Full'
        ).to(DEVICE)
        param_count = model.param_num()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=BEST_PARAMS['lr'],
            weight_decay=BEST_PARAMS['weight_decay']
        )
        steps_per_epoch = len(data_dict['train_loader'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=BEST_PARAMS['lr'],
            epochs=EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.2,
            anneal_strategy='cos'
        )
        best_val_mae = float('inf')
        best_model_state = None
        patience_counter = 0
        history_train_loss = []
        history_val_mae = []
        epoch_train_times = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(DEVICE)
        for epoch in range(EPOCHS):
            epoch_start_time = time.time()
            model.train()
            train_loss = []
            for bx, by in data_dict['train_loader'].loader:
                bx = bx.to(DEVICE).transpose(1, 3)
                by = by.to(DEVICE).transpose(1, 3)[:, :2, :, :]

                optimizer.zero_grad()

                output, _ = model(bx)

                pred = scaler.inverse_transform(
                    output.transpose(1, 3)
                ).transpose(1, 3)

                real = scaler.inverse_transform(
                    by.transpose(1, 3)
                ).transpose(1, 3)

                loss = utiln.masked_mae_loss(
                    pred,
                    real,
                    null_val=0.0
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )
                optimizer.step()
                scheduler.step()
                train_loss.append(loss.item())
            epoch_train_times.append(
                time.time() - epoch_start_time
            )
            model.eval()
            val_maes = []
            val_pk_maes = []
            val_dp_maes = []
            with torch.no_grad():
                for bx, by in data_dict['val_loader'].loader:
                    bx = bx.to(DEVICE).transpose(1, 3)
                    by = by.to(DEVICE).transpose(1, 3)[:, :2, :, :]
                    output, _ = model(bx)
                    pred = scaler.inverse_transform(
                        output.transpose(1, 3)
                    )
                    real = scaler.inverse_transform(
                        by.transpose(1, 3)
                    )
                    val_maes.append(
                        to_float(
                            utiln.MAE_torch(
                                pred,
                                real,
                                0.0
                            )
                        )
                    )
                    val_pk_maes.append(
                        to_float(
                            utiln.MAE_torch(
                                pred[..., 0],
                                real[..., 0],
                                0.0
                            )
                        )
                    )

                    val_dp_maes.append(
                        to_float(
                            utiln.MAE_torch(
                                pred[..., 1],
                                real[..., 1],
                                0.0
                            )
                        )
                    )

            current_val_mae = np.mean(val_maes)
            current_train_loss = np.mean(train_loss)

            print(
                f"[Seed {seed}] "
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Train Loss: {current_train_loss:.4f} | "
                f"Val MAE: {current_val_mae:.4f} "
                f"(Pick-up: {np.mean(val_pk_maes):.4f}, "
                f"Drop-off: {np.mean(val_dp_maes):.4f})"
            )
            history_train_loss.append(current_train_loss)
            history_val_mae.append(current_val_mae)
            if current_val_mae < best_val_mae:
                best_val_mae = current_val_mae
                best_model_state = copy.deepcopy(
                    model.state_dict()
                )
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= PATIENCE:
                print(
                    f"🛑 Seed {seed} 触发早停，"
                    f"最佳 Val MAE: {best_val_mae:.4f}"
                )
                break
        model_path = os.path.join(
            RESULT_DIR,
            f"best_model_seed_{seed}.pth"
        )
        torch.save(best_model_state, model_path)
        if torch.cuda.is_available():
            max_gpu_memory_gb = (
                torch.cuda.max_memory_allocated(DEVICE)
                / (1024 ** 3)
            )
        else:
            max_gpu_memory_gb = 0.0
        print(f"\n📊 开始测试 Seed {seed} 的最佳模型...")
        model.load_state_dict(best_model_state)
        model.eval()
        preds = []
        trues = []
        infer_start_time = time.time()
        with torch.no_grad():
            for bx, by in data_dict['test_loader'].loader:

                bx = bx.to(DEVICE).transpose(1, 3)
                by = by.to(DEVICE).transpose(1, 3)[:, :2, :, :]

                output, _ = model(bx)

                pred = scaler.inverse_transform(
                    output.transpose(1, 3)
                ).cpu()

                true = scaler.inverse_transform(
                    by.transpose(1, 3)
                ).cpu()

                preds.append(pred)
                trues.append(true)

        inference_time = time.time() - infer_start_time

        pred_cat = torch.cat(preds, dim=0)
        true_cat = torch.cat(trues, dim=0)
        np.save(
            os.path.join(
                RESULT_DIR,
                f"predicted_seed_{seed}.npy"
            ),
            pred_cat.numpy()
        )

        np.save(
            os.path.join(
                RESULT_DIR,
                f"true_test_seed_{seed}.npy"
            ),
            true_cat.numpy()
        )

        metrics = calculate_metrics(
            pred_cat,
            true_cat
        )

        avg_train_time_per_epoch = (
            np.mean(epoch_train_times)
            if epoch_train_times
            else 0.0
        )

        record = {
            'Seed': seed,
            'Result_Type': 'Single_Seed',

            'Params(M)': round(float(param_count), 4),
            'Train_Time/Epoch(s)': round(
                float(avg_train_time_per_epoch),
                2
            ),
            'Inference_Time(s)': round(
                float(inference_time),
                2
            ),
            'GPU_Mem(GB)': round(
                float(max_gpu_memory_gb),
                2
            )
        }

        for key, value in metrics.items():
            record[key] = round(float(value), 4)
        all_seed_records.append(record)

        print("\n" + "-" * 70)
        print(f"✅ Seed {seed} 测试完成")
        print("-" * 70)
        print(
            f"📊 [Global]   "
            f"MAE: {metrics['All_MAE']:.4f} | "
            f"RMSE: {metrics['All_RMSE']:.4f} | "
            f"MAPE: {metrics['All_MAPE(%)']:.4f}%"
        )
        print(
            f"🚖 [Pick-up]  "
            f"MAE: {metrics['Pick_MAE']:.4f} | "
            f"RMSE: {metrics['Pick_RMSE']:.4f} | "
            f"MAPE: {metrics['Pick_MAPE(%)']:.4f}%"
        )
        print(
            f"🚘 [Drop-off] "
            f"MAE: {metrics['Drop_MAE']:.4f} | "
            f"RMSE: {metrics['Drop_RMSE']:.4f} | "
            f"MAPE: {metrics['Drop_MAPE(%)']:.4f}%"
        )
        print(
            f"⏱️ Train/Epoch: {avg_train_time_per_epoch:.2f}s | "
            f"Inference: {inference_time:.2f}s | "
            f"GPU: {max_gpu_memory_gb:.2f} GB"
        )
        del model
        del optimizer
        del scheduler
        del best_model_state
        del pred_cat
        del true_cat
        del preds
        del trues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    df_single = pd.DataFrame(all_seed_records)
    excluded_columns = ['Seed', 'Result_Type']
    metric_columns = [
        column for column in df_single.columns
        if column not in excluded_columns
        and pd.api.types.is_numeric_dtype(df_single[column])
    ]

    mean_record = {
        'Seed': 'Mean',
        'Result_Type': 'Mean'
    }
    std_record = {
        'Seed': 'Std',
        'Result_Type': 'Std'
    }
    for column in metric_columns:
        mean_record[column] = round(
            float(df_single[column].mean()),
            4
        )
        std_record[column] = round(
            float(df_single[column].std(ddof=0)),
            4
        )
    df_summary = pd.DataFrame(
        [mean_record, std_record]
    )
    df_final = pd.concat(
        [
            df_single,
            df_summary
        ],
        ignore_index=True
    )
    df_final.to_csv(
        CSV_FILE,
        index=False,
        encoding='utf-8-sig'
    )
    print("\n\n" + "=" * 80)
    print(f"📈 {DATA_NAME} 多随机种子实验最终结果")
    print("=" * 80)
    display_columns = [
        'Seed',
        'Result_Type',
        'All_MAE',
        'All_RMSE',
        'All_MAPE(%)',
        'Pick_MAE',
        'Pick_RMSE',
        'Pick_MAPE(%)',
        'Drop_MAE',
        'Drop_RMSE',
        'Drop_MAPE(%)'
    ]

    print(
        df_final[display_columns].to_string(
            index=False
        )
    )
    mean_values = mean_record
    std_values = std_record
    print("\n" + "=" * 80)
    print("📝 论文结果：Mean ± Std")
    print("=" * 80)
    print(
        f"Global MAE:  "
        f"{mean_values['All_MAE']:.4f} ± "
        f"{std_values['All_MAE']:.4f}"
    )
    print(
        f"Global RMSE: "
        f"{mean_values['All_RMSE']:.4f} ± "
        f"{std_values['All_RMSE']:.4f}"
    )
    print(
        f"Global MAPE: "
        f"{mean_values['All_MAPE(%)']:.4f}% ± "
        f"{std_values['All_MAPE(%)']:.4f}%"
    )
    print(
        f"Pick-up MAE: "
        f"{mean_values['Pick_MAE']:.4f} ± "
        f"{std_values['Pick_MAE']:.4f}"
    )
    print(
        f"Pick-up RMSE: "
        f"{mean_values['Pick_RMSE']:.4f} ± "
        f"{std_values['Pick_RMSE']:.4f}"
    )
    print(
        f"Pick-up MAPE: "
        f"{mean_values['Pick_MAPE(%)']:.4f}% ± "
        f"{std_values['Pick_MAPE(%)']:.4f}%"
    )

    print(
        f"Drop-off MAE: "
        f"{mean_values['Drop_MAE']:.4f} ± "
        f"{std_values['Drop_MAE']:.4f}"
    )

    print(
        f"Drop-off RMSE: "
        f"{mean_values['Drop_RMSE']:.4f} ± "
        f"{std_values['Drop_RMSE']:.4f}"
    )

    print(
        f"Drop-off MAPE: "
        f"{mean_values['Drop_MAPE(%)']:.4f}% ± "
        f"{std_values['Drop_MAPE(%)']:.4f}%"
    )

    print("\n" + "=" * 80)
    print(f"✅ 多随机种子实验全部完成，共运行 {len(SEEDS)} 个随机种子")
    print(f"✅ 结果已保存至: {CSV_FILE}")
    print("=" * 80)
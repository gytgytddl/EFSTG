import h5py
import numpy as np
import os
import argparse


def add_nyc_time_features(data, steps_per_day=48):
    """
    为NYC数据添加时间特征。
    NYC数据 30min 一个步长，一天 48 个点。
    """
    num_samples, num_nodes, _ = data.shape
    time_indices = np.arange(num_samples)

    time_of_day = (time_indices % steps_per_day) / (steps_per_day - 1.0)
    day_of_week = ((time_indices // steps_per_day) % 7) / 6.0

    time_of_day_exp = np.tile(time_of_day.reshape(-1, 1, 1), (1, num_nodes, 1))
    day_of_week_exp = np.tile(day_of_week.reshape(-1, 1, 1), (1, num_nodes, 1))

    data_with_time = np.concatenate([data, day_of_week_exp, time_of_day_exp], axis=-1)
    return data_with_time


def generate_seq2seq_data(data, x_offsets, y_offsets):
    num_samples, num_nodes, _ = data.shape
    x, y = [], []

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))

    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    # Y 只保留前两个通道 [pick, drop]，不需要预测时间嵌入
    y = y[..., :2]
    return x, y


def process_nyc_data(file_path, output_dir, seq_len=12):
    print(f"\n开始处理: {os.path.basename(file_path)}")

    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        pick_key = [k for k in keys if 'pick' in k][0]
        drop_key = [k for k in keys if 'drop' in k][0]

        pick_data = f[pick_key][:]
        drop_data = f[drop_key][:]

    data = np.stack([pick_data, drop_data], axis=-1).astype(np.float32)

    data_with_time = add_nyc_time_features(data, steps_per_day=48)

    x_offsets = np.arange(-(seq_len - 1), 1, 1)
    y_offsets = np.arange(1, (seq_len + 1), 1)
    x, y = generate_seq2seq_data(data_with_time, x_offsets, y_offsets)

    print(f"样本生成完毕. X shape: {x.shape}, Y shape: {y.shape}")

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_train - num_test

    splits = {
        'train': (x[:num_train], y[:num_train]),
        'val': (x[num_train:num_train + num_val], y[num_train:num_train + num_val]),
        'test': (x[num_train + num_val:], y[num_train + num_val:])
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for cat, (x_s, y_s) in splits.items():
        save_path = os.path.join(output_dir, f"{cat}.npz")
        np.savez_compressed(save_path, x=x_s, y=y_s, x_offsets=x_offsets, y_offsets=y_offsets)
        print(f"已保存 {cat} 至 {save_path}")


if __name__ == '__main__':
    base_data_dir = './data'

    process_nyc_data(
        file_path=os.path.join(base_data_dir, 'nyc-bike.h5'),
        output_dir=os.path.join(base_data_dir, 'processed/NYC-BIKE'),
        seq_len=12
    )

    process_nyc_data(
        file_path=os.path.join(base_data_dir, 'nyc-taxi.h5'),
        output_dir=os.path.join(base_data_dir, 'processed/NYC-TAXI'),
        seq_len=12
    )
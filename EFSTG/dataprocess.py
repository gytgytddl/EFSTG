import argparse
import numpy as np
import os
import pandas as pd


def add_time_features(data):
    num_samples, num_nodes, _ = data.shape
    time_indices = np.arange(num_samples)
    time_of_day = (time_indices % 288) / 287.0
    day_of_week = ((time_indices // 288) % 7) / 6.0

    time_of_day = time_of_day.reshape(-1, 1)
    day_of_week = day_of_week.reshape(-1, 1)
    time_of_day_expanded = np.tile(time_of_day, (1, num_nodes)).reshape(num_samples, num_nodes, 1)
    day_of_week_expanded = np.tile(day_of_week, (1, num_nodes)).reshape(num_samples, num_nodes, 1)
    data_with_time = np.concatenate([data, day_of_week_expanded, time_of_day_expanded], axis=-1)
    print(f"原始数据形状: {data.shape}")
    print(f"添加时间特征后形状: {data_with_time.shape}")
    return data_with_time
def generate_graph_seq2seq_io_data(
        data, x_offsets, y_offsets
):
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
    y = y[..., 0:1]
    return x, y
def generate_train_val_test(args):
    data_raw = np.load(args.traffic_df_filename)['data']
    data_occupancy = data_raw[:, :, 1:2]
    data_with_time = add_time_features(data_occupancy)

    seq_length_x, seq_length_y = args.seq_length_x, args.seq_length_y
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)
    y_offsets = np.arange(args.y_start, (seq_length_y + 1), 1)

    x, y = generate_graph_seq2seq_io_data(data=data_with_time, x_offsets=x_offsets, y_offsets=y_offsets)

    print("x shape: ", x.shape, ", y shape: ", y.shape)

    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.6)
    num_val = num_samples - num_train - num_test

    x_train, y_train = x[:num_train], y[:num_train]

    x_val, y_val = x[num_train:num_train + num_val], y[num_train:num_train + num_val]

    x_test, y_test = x[num_train + num_val:], y[num_train + num_val:]

    for cat in ['train', 'val', 'test']:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]

        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="data/processed/PEMS07/", help="输出文件夹")
    parser.add_argument('--traffic_df_filename', type=str, default="data/PEMS07/PEMS07.npz", help="数据集")
    parser.add_argument('--seq_length_x', type=int, default=12, help='输入序列长度')
    parser.add_argument('--seq_length_y', type=int, default=12, help='输出序列长度')
    parser.add_argument('--y_start', type=int, default=1, help='从第几天开始预测')

    args = parser.parse_args()

    if os.path.exists(args.output_dir):
        reply = str(input(f'{args.output_dir} 存在，是否将其作为输出目录?(y/n)')).lower().strip()
        if reply[0] != 'y':
            exit()
    else:
        os.makedirs(args.output_dir)

    generate_train_val_test(args)

"""
PEMS04:
x shape:  (16969, 12, 307, 1) , y shape:  (16969, 12, 307, 1)
train x:  (10181, 12, 307, 1) y: (10181, 12, 307, 1)
val x:  (3394, 12, 307, 1) y: (3394, 12, 307, 1)
test x:  (3394, 12, 307, 1) y: (3394, 12, 307, 1)

PEMSO8:
x shape:  (17833, 12, 170, 1) , y shape:  (17833, 12, 170, 1)
train x:  (10700, 12, 170, 1) y: (10700, 12, 170, 1)
val x:  (3566, 12, 170, 1) y: (3566, 12, 170, 1)
test x:  (3567, 12, 170, 1) y: (3567, 12, 170, 1) PEMS07	883
"""



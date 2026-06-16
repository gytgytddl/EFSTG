import numpy as np
import pandas as pd
import networkx as nx
import os


def create_pems_distance_matrix(adj_csv_path, num_nodes, output_path):
    print("=" * 50)
    print(f"开始为 PEMS 类型数据创建距离矩阵...")
    print(f"加载边数据: {adj_csv_path}")

    try:
        edges_df = pd.read_csv(adj_csv_path)
    except FileNotFoundError:
        print(f"错误: 文件未找到 {adj_csv_path}，请检查路径。")
        return

    all_node_ids = set(edges_df['from'].unique()) | set(edges_df['to'].unique())
    if len(all_node_ids) > num_nodes:
        print(f"警告: 文件中的节点ID数量 ({len(all_node_ids)}) 超出了预期的节点数 ({num_nodes})。")

    print("构建图中...")
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(int(row['from']), int(row['to']), weight=row['cost'])

    print("正在使用 Dijkstra 算法计算所有节点对之间的最短路径...")
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(G))

    distance_matrix = np.full((num_nodes, num_nodes), np.inf)
    for i in range(num_nodes):
        if i in path_lengths:
            for j, dist in path_lengths[i].items():
                distance_matrix[i, j] = dist

    print("距离矩阵计算完成。")

    if np.isinf(distance_matrix).any():
        print("警告: 图存在不连通的部分，某些节点间距离为无穷大。")
        max_dist = np.nanmax(distance_matrix[distance_matrix != np.inf])
        if np.isnan(max_dist):
            max_dist = 1.0
        distance_matrix[np.isinf(distance_matrix)] = max_dist * 2 + 1
        print("已将无穷大距离替换为大数值。")

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    np.save(output_path, distance_matrix)
    print(f"距离矩阵 (形状: {distance_matrix.shape}) 已成功保存至 '{output_path}'")
    print("=" * 50)


if __name__ == '__main__':
    ADJ_CSV_PATH = 'data/PEMS08/PEMS08.csv'
    NUM_NODES = 170
    OUTPUT_PATH = 'data/processed/PEMS08/pems08_distance_matrix.npy'

    create_pems_distance_matrix(ADJ_CSV_PATH, NUM_NODES, OUTPUT_PATH)

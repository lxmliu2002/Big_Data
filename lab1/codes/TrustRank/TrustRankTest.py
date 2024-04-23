import time
import numpy as np
import os
import psutil
import goodWebSearch

def read_graph(file_path):
    # 读取图数据并构建图结构
    G = {}
    nodes = []
    with open(file_path, 'r') as f:
        for line in f:
            from_node, to_node = map(int, line.strip().split())
            if from_node not in G:
                G[from_node] = []
            G[from_node].append(to_node)
            if from_node not in nodes:
                nodes.append(from_node)
            if to_node not in nodes:
                nodes.append(to_node)
    return G, nodes

def read_seed_vector(file_path):
    # 读取种子向量
    seed_vector = []
    with open(file_path, 'r') as f:
        for line in f:
            node = int(line.strip())
            seed_vector.append(node)
    return seed_vector

def calculate_trustrank_and_sort(G, nodes, N, teleport_parameter, seed_vector):
    """
    迭代法求解TrustRank并排序

    Parameters:
        G (dict): 图的邻接表表示，键为节点，值为与该节点相连的节点列表
        nodes (list): 图中所有节点的列表
        N (int): 图中节点的总数
        teleport_parameter (float): 随机跳转参数，控制随机跳转的概率
        seed_vector (list): 初始种子向量

    Returns:
        list: 按TrustRank值从高到低排序的节点列表，包含节点和对应的TrustRank值
    """
    start = time.perf_counter()

    # 创建节点索引字典
    index = {}
    for i, node in enumerate(sorted(nodes)):
        index[node] = i

    # 创建转移矩阵S
    S = np.zeros([N, N], dtype=np.float64)
    for from_node, to_nodes in G.items():
        for to_node in to_nodes:
            S[index[to_node], index[from_node]] = 1

    # 对转移矩阵S进行归一化处理
    for j in range(N):
        sum_of_col = sum(S[:, j])
        if sum_of_col == 0:
            S[:, j] = 1/N
        else:
            S[:, j] /= sum_of_col

    # 初始化TrustRank向量T_n
    T_n = np.zeros(N, dtype=np.float64)
    for node in seed_vector:
        T_n[index[node]] = 1 / len(seed_vector)

    D = T_n

    # 构建TrustRank迭代矩阵A
    A = teleport_parameter * S

    # 设置迭代停止条件
    e = 100
    tol = 1 / (N * N)

    # 迭代计算TrustRank
    while e > tol:
        T_n1 = np.dot(A, T_n)  + (1 - teleport_parameter) * D
        e = T_n1 - T_n
        e = max(map(abs, e))
        T_n = T_n1

    # 根据TrustRank值对节点进行排序
    sorted_nodes = sorted(index.items(), key=lambda x: T_n[x[1]], reverse=True)

    # 提取前100个排序结果
    sorted_results = []
    for node, index in sorted_nodes[:len(sorted_nodes)]:
        sorted_results.append((node, T_n[index]))
    end = time.perf_counter()
    print(u'当前进程的内存使用：%.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    print('当前进程的时间开销：%.4f s' % (end - start))
    return sorted_results


def output_result(results, file_path):
    # 将结果输出到文件
    with open(file_path, 'w') as f:
        for node, score in results:
            f.write(f"{node} {score}\n")


def TR():
    """
    迭代法求解TrustRank并输出结果
    """
    goodWebSearch.Search()
    data_file_path = '../Data.txt'
    seed_vector_file_path = 'good.txt'
    output_file_path = './compare/trustrank.txt'

    teleport_parameter = 0.85

    G, nodes = read_graph(data_file_path)
    seed_vector = read_seed_vector(seed_vector_file_path)
    N = len(nodes)
    sorted_results = calculate_trustrank_and_sort(G, nodes, N, teleport_parameter, seed_vector)
    output_result(sorted_results, output_file_path)


TR()

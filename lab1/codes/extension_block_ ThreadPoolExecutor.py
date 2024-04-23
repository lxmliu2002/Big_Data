import concurrent.futures
import time

import numpy as np
import psutil
import os


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


def iter_once(G, P_n, start,end, teleport_parameter,N):
    P_n1 = np.zeros(end-start, dtype=np.float64)
    for i in range(N):
        if i in G.keys():
            exist = np.isin(np.arange(start,end), G[i])
            P_n1 += ((teleport_parameter * exist / len(G[i]) + (1 - teleport_parameter) / N)) * P_n[i]
        else:
            P_n1 += P_n[i] / N
    return P_n1


def calculate_pagerank_and_sort(G, nodes, N, teleport_parameter, block_size):
    """
    分块更新迭代法求解PageRank并排序

    Parameters:
        G (dict): 图的邻接表表示，键为节点，值为与该节点相连的节点列表
        nodes (list): 图中所有节点的列表
        N (int): 图中节点的总数
        teleport_parameter (float): 随机跳转参数，控制随机跳转的概率
        block_size (int): 每个块的大小

    Returns:
        list: 按PageRank值从高到低排序的节点列表，包含节点和对应的PageRank值
    """
    start = time.perf_counter()
    # 创建节点索引字典
    index = {}
    for i, node in enumerate(sorted(nodes)):
        index[node] = i

    # 将图中key和value转为对应的索引
    G = {index[k]: [index[v] for v in vs] for k, vs in G.items()}

    # 初始化PageRank向量P_n
    P_n = np.ones(N, dtype=np.float64) / N

    # 设置迭代停止条件
    e = 100
    tol = 1 / (N * N)

    # 迭代计算PageRank
    while e > tol:
        # print(e)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for block_start in range(0, N, block_size):
                block_end = min(block_start + block_size, N)
                futures.append(executor.submit(iter_once, G, P_n, block_start, block_end, teleport_parameter, N))
            P_n1 = np.concatenate([future.result() for future in futures])
        e = P_n1 - P_n
        e = max(map(abs, e))
        P_n = P_n1

    # 根据PageRank值对节点进行排序
    sorted_nodes = sorted(index.items(), key=lambda x: P_n[x[1]], reverse=True)

    # 提取前100个排序结果
    sorted_results = []
    for node, index in sorted_nodes[:100]:
        sorted_results.append((node, P_n[index]))

    end = time.perf_counter()
    print(u'当前进程的内存使用：%.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    print('当前进程的时间开销：%.4f s' % (end - start))

    return sorted_results


def output_result(results, file_path):
    # 将结果输出到文件
    with open(file_path, 'w') as f:
        for node, score in results:
            f.write(f"{node} {score}\n")


if __name__ == '__main__':
    """
    分块更新迭代法求解PageRank并输出结果
    """
    data_file_path = 'Data.txt'
    output_file_path = 'output/block_ThreadPoolExecutor.txt'

    teleport_parameter = 0.85
    block_size = 100

    G, nodes = read_graph(data_file_path)
    N = len(nodes)
    sorted_results = calculate_pagerank_and_sort(G, nodes, N, teleport_parameter, block_size)
    output_result(sorted_results, output_file_path)

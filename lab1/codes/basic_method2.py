import numpy as np
import os
import psutil
import time


def read_graph(file_path):
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


def calculate_pagerank_and_sort_sum(G, nodes, N, teleport_parameter):
    """
    代数法求解
    """
    start = time.perf_counter()
    index = {}
    for i, node in enumerate(sorted(nodes)):
        index[node] = i

    S = np.zeros([N, N], dtype=np.float64)
    for from_node, to_nodes in G.items():
        for to_node in to_nodes:
            S[index[to_node], index[from_node]] = 1

    for j in range(N):
        sum_of_col = sum(S[:, j])
        if sum_of_col == 0:
            S[:, j] = 1 / N
        else:
            S[:, j] /= sum_of_col

    e = np.identity(N, dtype=np.float64)
    eT = np.ones([N, 1], dtype=np.float64)

    P = np.dot(np.linalg.inv(e - teleport_parameter * S), ((1 - teleport_parameter) / N * eT)).flatten()

    sorted_nodes = sorted(index.items(), key=lambda x: P[x[1]], reverse=True)

    sorted_results = []
    for node, index in sorted_nodes[:100]:
        sorted_results.append((node, P[index]))

    end = time.perf_counter()
    print(u'当前进程的内存使用：%.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    print('当前进程的时间开销：%.4f s' % (end - start))

    return sorted_results


def output_result(results, file_path):
    with open(file_path, 'w') as f:
        for node, score in results:
            f.write(f"{node} {score}\n")


if __name__ == '__main__':
    """
    代数法求解
    """
    data_file_path = './Data.txt'
    output_file_path = './output/basic_2.txt'

    teleport_parameter = 0.85

    G, nodes = read_graph(data_file_path)
    N = len(nodes)
    sorted_results = calculate_pagerank_and_sort_sum(G, nodes, N, teleport_parameter)
    output_result(sorted_results, output_file_path)
import os
import time

import networkx as nx
import psutil


def read_graph(file_path):
    # 创建一个有向图对象
    G = nx.DiGraph()
    with open(file_path, 'r') as f:
        for line in f:
            # 从文件中读取每一行，并将起始节点和目标节点转换为整数
            from_node, to_node = map(int, line.strip().split())
            # 在图中添加边
            G.add_edge(from_node, to_node)
    return G


def calculate_pagerank_and_sort(G, teleport_parameter=0.85, tol=1e-20, top_n=100):
    start = time.perf_counter()

    # 使用PageRank算法计算节点的PageRank得分
    pagerank_scores = nx.pagerank(G, alpha=teleport_parameter, tol=tol)
    # 将得分进行排序
    sorted_scores = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

    end = time.perf_counter()
    print(u'当前进程的内存使用：%.4f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    print('当前进程的时间开销：%.4f s' % (end - start))
    return sorted_scores[:top_n]


def output_result(results, file_path):
    with open(file_path, 'w') as f:
        for node, score in results:
            # 将节点和得分写入文件
            f.write(f"{node} {score}\n")


if __name__ == '__main__':
    data_file_path = './Data.txt'
    output_file_path = './output/networkx.txt'

    teleport_parameter = 0.85
    # tol = 1e-10

    G = read_graph(data_file_path)
    # 计算tol的值
    tol = 1 / (G.number_of_nodes() * G.number_of_nodes())

    sorted_results = calculate_pagerank_and_sort(G=G, teleport_parameter=teleport_parameter, tol=tol, top_n=100)
    # 将结果输出到文件
    output_result(sorted_results, output_file_path)

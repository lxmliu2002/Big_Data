import networkx as nx

import TrustRankTest

for i in range(10):

    TrustRankTest.TR()

    # 从test2.txt文件中读取节点
    with open('test2.txt', 'r') as file:
        nodes = file.read().splitlines()

    # 计算每个节点的排名变化
    ranking_changes = []
    for node in nodes:
        with open('./compare/networkx.txt', 'r') as networkx_file, open('./output/trustrank.txt', 'r') as trustrank_file:
            networkx_ranking = [line.split()[0] for line in networkx_file]
            trustrank_ranking = [line.split()[0] for line in trustrank_file]

        networkx_index = networkx_ranking.index(node)
        trustrank_index = trustrank_ranking.index(node)

        ranking_change = trustrank_index - networkx_index
        ranking_changes.append(ranking_change)

    # 计算平均排名变化
    average_change = sum(ranking_changes) / len(ranking_changes)

    # 打印平均排名变化
    print("Average ranking change:", average_change)

    # 将排名变化写入result.txt文件
    with open('result.txt', 'w') as result_file:
        for change in ranking_changes:
            result_file.write(str(change) + '\n')

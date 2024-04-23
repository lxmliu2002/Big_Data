import random

def read_graph(file_path):
    # 读取图数据并构建图结构
    G = {}
    nodes = []
    in_degrees = {}
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
            if to_node not in in_degrees:
                in_degrees[to_node] = 0
            in_degrees[to_node] += 1
    return G, nodes, in_degrees


def output_result(results, file_path, in_degrees):
    # 将结果输出到文件
    with open(file_path, 'w') as f:
        for node in results:
            f.write(f"{node}\n")


def find_low_in_degree_nodes(G, nodes, in_degrees, target_nodes, num_nodes):
    # 找到与目标节点不相邻且入度较小的节点
    low_in_degree_nodes = []
    for node in nodes:
        if node not in target_nodes:
            is_adjacent = False
            for target_node in target_nodes:
                if target_node in G and node in G[target_node]:
                    is_adjacent = True
                    break
            if not is_adjacent and in_degrees.get(node, 0) < 9:
                low_in_degree_nodes.append(node)
    random.shuffle(low_in_degree_nodes)
    selected_nodes = low_in_degree_nodes[:num_nodes]
    return selected_nodes


def Search():
    data_file_path = '../Data.txt'
    output_file_path = 'good.txt'
    output_file_path2 = 'bad.txt'

    G, nodes, in_degrees = read_graph(data_file_path)

    # 找到入度最高的前5%的节点
    sorted_nodes = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
    top_5_percent = int(len(sorted_nodes) * 0.05)
    top_nodes = [node for node, _ in sorted_nodes[:top_5_percent]]

    # 随机选择100个节点
    random.shuffle(top_nodes)
    selected_nodes = top_nodes[:100]

    # 将选定的节点存储在数组中
    selected_nodes_array = []
    for node in selected_nodes:
        selected_nodes_array.append(node)

    # 将选定的节点输出到test.txt文件
    output_result(selected_nodes_array, output_file_path, in_degrees)

    # 找到与目标节点不直接相邻且入度较小的节点，并输出到test2.txt文件
    selected_nodes2 = find_low_in_degree_nodes(G, nodes, in_degrees, selected_nodes, 100)
    output_result(selected_nodes2, output_file_path2, in_degrees)


Search()
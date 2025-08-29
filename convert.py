# FILE: preprocess_quantum_datasets_minimal.py
import os
import json
import numpy as np
import math
import networkx as nx
from collections import deque

ROOT_PATH = r'C:\Users\59415\Desktop\Pioneer RI\Research\Code\datasets'


def fix_tree_neighbors_ordering(G, root=0):
    """为树结构排序邻居: 非根节点[父节点, 子节点...], 根节点[子节点...]"""
    parent_map, child_map = {}, {node: [] for node in G.nodes}
    visited, queue = set([root]), deque([root])

    while queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                parent_map[neighbor] = node
                child_map[node].append(neighbor)
                queue.append(neighbor)

    return [
        [parent_map[node]] + sorted(child_map[node]) if node != root else sorted(child_map[node])
        for node in sorted(G.nodes)
    ]


def identify_glued_tree_parts(G):
    """识别Glued Tree中的两个子树和连接边"""
    connection_edges = []
    for u, v in G.edges():
        if not (nx.has_path(G.subgraph(G.nodes - {u}), 0, v) or
                nx.has_path(G.subgraph(G.nodes - {v}), 0, u)):
            connection_edges.append((u, v))

    tree1_nodes = set(nx.node_connected_component(G, 0))
    tree2_nodes = set(G.nodes) - tree1_nodes

    return tree1_nodes, tree2_nodes, connection_edges


def get_glued_tree_target_node(G):
    """找Glued Tree中远端树的根节点"""
    tree1_nodes, tree2_nodes, _ = identify_glued_tree_parts(G)
    target_tree = tree2_nodes if 0 in tree1_nodes else tree1_nodes
    G_target = G.subgraph(target_tree)

    # 找目标树的中心节点
    central_node = min(target_tree, key=lambda n: nx.eccentricity(G_target, n))
    return central_node


def calculate_qubit_requirements(G, structure_type, data):
    """计算各量子算法所需量子比特"""
    N, deg = G.number_of_nodes(), max(dict(G.degree()).values()) if G else 1
    pos_qubits = max(1, int(math.ceil(math.log2(N))))

    return {
        'DTQW': {
            'num_position_qubits': pos_qubits,
            'num_coin_qubits': max(1, int(math.ceil(math.log2(deg))))
        },
        'TreeInfo': {
            'theoretical_dtqw_hit': math.pi * math.sqrt(data.get('branch', 1) ** data.get('height', 0)) / 2
        } if structure_type == 'Tree' else {}
    }


def determine_target_node(G, structure_type, data):
    """确定目标节点"""
    if structure_type == 'GluedTree':
        try:
            return get_glued_tree_target_node(G)
        except:
            return max(G.nodes)

    if structure_type == 'Tree':
        branch, height = data.get('branch'), data.get('height')
        if branch and height:
            leaf_start = ((branch ** height) - 1) // (branch - 1) if branch > 1 else 0
            return min(G.nodes - 1, leaf_start + (branch ** (height - 1)))

    try:
        return max(nx.single_source_shortest_path_length(G, 0).items(), key=lambda x: x[1])[0]
    except:
        return G.number_of_nodes() - 1


def preprocess_single_graph(graph_data, structure_type):
    """处理单个图"""
    data = graph_data.copy()
    data['structure_type'] = structure_type

    # 构建图
    G = nx.from_numpy_array(np.array(data['adj_matrix']))

    # 处理树邻居顺序
    if structure_type == 'Tree' and nx.is_connected(G) and nx.is_tree(G):
        data['neighbors'] = fix_tree_neighbors_ordering(G)
    else:
        data['neighbors'] = [list(G.neighbors(i)) for i in range(len(G.nodes))]

    # 设置目标节点
    data['target_node'] = determine_target_node(G, structure_type, data)

    # 添加量子信息
    data['quantum_info'] = {
        'qubits': calculate_qubit_requirements(G, structure_type, data),
        'num_position_qubits': max(1, int(math.ceil(math.log2(G.number_of_nodes())))),
        'start_node': 0
    }

    return data


def main():
    for struct_type in ['Tree', 'Grid', 'SmallWorld', 'GluedTree', 'Hypercube']:
        dataset_dir = os.path.join(ROOT_PATH, f'datasets-{struct_type.lower()}s')
        if not os.path.exists(dataset_dir): continue

        # 处理所有JSON文件
        for filename in os.listdir(dataset_dir):
            if not filename.endswith('.json') or '_quantum' in filename:
                continue

            filepath = os.path.join(dataset_dir, filename)

            with open(filepath, 'r') as f:
                data = json.load(f)

            # 处理单个图或图列表
            results = (
                [preprocess_single_graph(g, struct_type) for g in data]
                if isinstance(data, list)
                else [preprocess_single_graph(data, struct_type)]
            )

            # 保存结果
            output_path = filepath.replace('.json', '_quantum.json')
            with open(output_path, 'w') as f:
                json.dump(results[0] if len(results) == 1 else results, f, indent=2)


if __name__ == '__main__':
    main()
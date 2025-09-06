import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
import os
import random


root_path = r'C:\Users\59415\Desktop\Pioneer RI\Research\Code\datasets'
os.makedirs(root_path, exist_ok=True)



def generate_and_save(graph_id, G, structure_type, seq_num):

    sub_dir = os.path.join(root_path, f'datasets-{structure_type.lower()}s')
    os.makedirs(sub_dir, exist_ok=True)


    filename = f'{structure_type.lower()}s_{seq_num:03d}_{graph_id}'


    adj_matrix = nx.to_numpy_array(G)


    csv_path = os.path.join(sub_dir, f'{filename}_adj.csv')
    np.savetxt(csv_path, adj_matrix, delimiter=',', fmt='%d')


    data = {
        'graph_id': graph_id,
        'structure_type': structure_type,
        'nodes': list(G.nodes()),
        'adj_matrix': adj_matrix.tolist()
    }
    json_path = os.path.join(sub_dir, f'{filename}.json')
    with open(json_path, 'w') as f:
        json.dump(data, f)


    plt.figure()
    nx.draw(G, with_labels=True)
    plt.title(f'{filename} - {structure_type}')
    plt.savefig(os.path.join(sub_dir, f'{filename}.png'))
    plt.close()

    print(f'Generated: {filename} (nodes: {G.number_of_nodes()}) in {sub_dir}')
    return filename, data



seq_counter = 1
all_variants = {'Tree': [], 'Grid': [], 'SmallWorld': [], 'GluedTree': [], 'Hypercube': []}
all_data = {'Tree': [], 'Grid': [], 'SmallWorld': [], 'GluedTree': [], 'Hypercube': []}


for branch in [2, 3, 4, 5]:
    for height in [2, 3, 4, 5]:
        if branch > 3 and height == 5: continue
        G = nx.balanced_tree(r=branch, h=height)
        graph_id = f'branch{branch}_height{height}'
        filename, data = generate_and_save(graph_id, G, 'Tree', seq_counter)
        all_variants['Tree'].append(filename)
        all_data['Tree'].append(data)
        seq_counter += 1


for size in [3, 4, 5, 6, 7, 8]:
    for density_var in [0.0, 0.2, 0.4]:
        G = nx.grid_2d_graph(size, size)
        num_edges = int(G.number_of_edges() * density_var)
        nodes_list = list(G.nodes())
        try:
            for _ in range(num_edges):
                idx = np.random.choice(len(nodes_list), 2, replace=False)
                u = nodes_list[idx[0]]
                v = nodes_list[idx[1]]
                if u != v and not G.has_edge(u, v):
                    G.add_edge(u, v)
        except Exception as e:
            print(f"Error in grid generation for size {size} density {density_var}: {e}")
            continue
        graph_id = f'size{size}_density{round(density_var, 2)}'
        filename, data = generate_and_save(graph_id, G, 'Grid', seq_counter)
        all_variants['Grid'].append(filename)
        all_data['Grid'].append(data)
        seq_counter += 1


for n in [10, 15, 20, 25, 30, 35]:
    for p in [0.1, 0.3, 0.6]:
        G = nx.watts_strogatz_graph(n=n, k=4, p=p)
        graph_id = f'n{n}_p{p}'
        filename, data = generate_and_save(graph_id, G, 'SmallWorld', seq_counter)
        all_variants['SmallWorld'].append(filename)
        all_data['SmallWorld'].append(data)
        seq_counter += 1


for height in [3, 4, 5]:
    for num_glues in [2, 4]:
        tree1 = nx.balanced_tree(r=2, h=height)
        tree2 = nx.balanced_tree(r=2, h=height)
        offset = tree1.number_of_nodes()
        tree2 = nx.relabel_nodes(tree2, {node: node + offset for node in tree2.nodes()})
        G = nx.union(tree1, tree2)
        leaves1 = [node for node in tree1.nodes() if tree1.degree(node) == 1]
        leaves2 = [node for node in tree2.nodes() if tree2.degree(node) == 1]
        if len(leaves1) < num_glues or len(leaves2) < num_glues:
            num_glues = min(len(leaves1), len(leaves2))
        glue_pairs = random.sample(list(zip(random.sample(leaves1, num_glues), random.sample(leaves2, num_glues))),
                                   num_glues)
        for u, v in glue_pairs:
            G.add_edge(u, v)
        graph_id = f'height{height}_glues{num_glues}'
        filename, data = generate_and_save(graph_id, G, 'GluedTree', seq_counter)
        all_variants['GluedTree'].append(filename)
        all_data['GluedTree'].append(data)
        seq_counter += 1


for dim in [3, 4, 5, 6]:
    for perturb in [0.0, 0.1]:
        G = nx.hypercube_graph(dim)
        num_nodes = 2 ** dim
        num_perturb_edges = int(num_nodes * perturb)
        nodes_list = list(G.nodes())
        try:
            for _ in range(num_perturb_edges):
                idx = np.random.choice(len(nodes_list), 2, replace=False)
                u = nodes_list[idx[0]]
                v = nodes_list[idx[1]]
                if u != v and not G.has_edge(u, v):
                    G.add_edge(u, v)
        except Exception as e:
            print(f"Error in hypercube generation for dim {dim} perturb {perturb}: {e}")
            continue
        graph_id = f'dim{dim}_perturb{round(perturb, 2)}'
        filename, data = generate_and_save(graph_id, G, 'Hypercube', seq_counter)
        all_variants['Hypercube'].append(filename)
        all_data['Hypercube'].append(data)
        seq_counter += 1


for structure_type in all_data:
    sub_dir = os.path.join(root_path, f'datasets-{structure_type.lower()}s')
    all_json_path = os.path.join(sub_dir, f'{structure_type.lower()}s_all_variants.json')
    with open(all_json_path, 'w') as f:
        json.dump(all_data[structure_type], f)
    print(f'Created overall file for {structure_type}: {all_json_path}')


for structure_type, variants in all_variants.items():
    sub_dir = os.path.join(root_path, f'datasets-{structure_type.lower()}s')
    summary_path = os.path.join(sub_dir, f'{structure_type.lower()}s_summary.csv')
    with open(summary_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Graph ID', 'Nodes'])
        for vid in variants:
            writer.writerow([vid, 'See JSON', 'See JSON'])


total_summary_path = os.path.join(root_path, 'dataset_overall_summary.csv')
with open(total_summary_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Type', 'Count', 'Filenames'])
    for typ, vars in all_variants.items():
        writer.writerow([typ, len(vars), ', '.join(vars[:3]) + '...'])

total_count = sum(len(v) for v in all_variants.values())
print(f'Total variants generated: {total_count}')
print('Datasets saved in subfolders under:', root_path)

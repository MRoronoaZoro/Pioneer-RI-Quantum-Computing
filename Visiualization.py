import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import binom
import os
import re


plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


root_dir = r'C:\Users\59415\Desktop'


colors = {
    'DTQW': '#1f77b4',
    'CTQW': '#ff7f0e',
    'Grover-QRW': '#2ca02c'
}


markers = {
    'DTQW': 'o',
    'CTQW': 's',
    'Grover-QRW': '^'
}

def parse_graph_info(graph_id):
    """解析graph_id获取图类型和节点数N"""
    if 'branch' in graph_id and 'height' in graph_id:

        parts = graph_id.split('_')
        branch = int(parts[0].replace('branch', ''))
        height = int(parts[1].replace('height', ''))
        N = (branch ** (height + 1) - 1) // (branch - 1) if branch > 1 else height + 1
        return 'Tree', N
    elif 'size' in graph_id:

        size = int(re.search(r'size(\d+)', graph_id).group(1))
        N = size ** 2
        return 'Grid', N
    elif graph_id.startswith('n') and 'p' in graph_id:

        n = int(re.search(r'n(\d+)', graph_id).group(1))
        return 'SmallWorld', n
    elif 'height' in graph_id and 'glues' in graph_id.lower():

        height = int(re.search(r'height(\d+)', graph_id).group(1))
        single_tree_nodes = (2 ** (height + 1) - 1)
        N = single_tree_nodes * 2
        return 'GluedTree', N
    elif 'dim' in graph_id or 'hypercube' in graph_id.lower():

        dim = int(re.search(r'dim(\d+)', graph_id).group(1))
        N = 2 ** dim
        return 'Hypercube', N
    else:
        return 'Unknown', 0

def binomial_ci(successes, n, confidence=0.95):
    """计算二项分布的置信区间"""
    if n == 0:
        return 0, 0
    p = successes / n
    alpha = 1 - confidence
    ci = binom.interval(confidence, n, p)
    return ci[0]/n, ci[1]/n

def load_and_process_data():
    """加载和处理数据"""

    dtqw_file = os.path.join(root_dir, 'DTQW_Analysis_Results_Fixed.csv')
    ctqw_file = os.path.join(root_dir, 'CTQW_Analysis_Results.csv')
    grover_file = os.path.join(root_dir, 'Grover_Analysis_Results_Fixed.csv')


    try:
        dtqw_df = pd.read_csv(dtqw_file, encoding='utf-8')
    except:
        dtqw_df = pd.read_csv(dtqw_file, encoding='gbk')

    try:
        ctqw_df = pd.read_csv(ctqw_file, encoding='utf-8')
    except:
        ctqw_df = pd.read_csv(ctqw_file, encoding='gbk')

    try:
        grover_df = pd.read_csv(grover_file, encoding='utf-8')
    except:
        grover_df = pd.read_csv(grover_file, encoding='gbk')


    column_mapping = {
        '数据名称': 'graph_id',
        '类型': 'type_orig',
        'avg_runtime': 'search_time'
    }

    for df in [dtqw_df, ctqw_df, grover_df]:
        df.rename(columns=column_mapping, inplace=True)

        if 'graph_id' not in df.columns and len(df.columns) > 0:
            df.rename(columns={df.columns[0]: 'graph_id'}, inplace=True)


    dtqw_df['algorithm'] = 'DTQW'
    ctqw_df['algorithm'] = 'CTQW'
    grover_df['algorithm'] = 'Grover-QRW'


    all_df = pd.concat([dtqw_df, ctqw_df, grover_df], ignore_index=True)


    graph_info = all_df['graph_id'].apply(parse_graph_info)
    all_df['type'] = graph_info.apply(lambda x: x[0])
    all_df['N'] = graph_info.apply(lambda x: x[1])


    valid_types = ['Tree', 'Grid', 'SmallWorld', 'GluedTree', 'Hypercube']
    all_df = all_df[all_df['type'].isin(valid_types)]


    all_df = all_df.sort_values(['type', 'N', 'graph_id'])

    return all_df

def create_visualizations(all_df):
    """创建所有可视化图表"""

    output_dir = os.path.join(root_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)


    if all_df.empty:
        print("警告：数据为空，无法生成图表")
        return

    print(f"数据形状: {all_df.shape}")
    print(f"图类型: {all_df['type'].unique()}")
    print(f"算法: {all_df['algorithm'].unique()}")


    if 'search_probability' in all_df.columns:
        plt.figure(figsize=(15, 10))


        types = all_df['type'].unique()
        n_types = len(types)

        if n_types > 0:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for i, graph_type in enumerate(types):
                if i < len(axes):
                    ax = axes[i]
                    type_data = all_df[all_df['type'] == graph_type]

                    if not type_data.empty:
                        sns.barplot(data=type_data, x='graph_id', y='search_probability',
                                  hue='algorithm', palette=colors, ax=ax)
                        ax.set_title(f'{graph_type} - Search Probability')
                        ax.set_xlabel('Graph ID')
                        ax.set_ylabel('Search Probability')
                        ax.tick_params(axis='x', rotation=45)




            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'figure1_search_probability_bar.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("图1已保存: Search Probability 柱状图")


    if 'search_time' in all_df.columns and len(all_df['type'].unique()) > 0:
        plt.figure(figsize=(15, 10))

        types = all_df['type'].unique()
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, graph_type in enumerate(types):
            if i < len(axes):
                ax = axes[i]
                type_data = all_df[all_df['type'] == graph_type]

                if not type_data.empty:
                    sns.barplot(data=type_data, x='graph_id', y='search_time',
                              hue='algorithm', palette=colors, ax=ax)
                    ax.set_title(f'{graph_type} - Search Time')
                    ax.set_xlabel('Graph ID')
                    ax.set_ylabel('Search Time (log scale)')
                    ax.set_yscale('log')
                    ax.tick_params(axis='x', rotation=45)


        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figure3_search_time_bar.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("图3已保存: Search Time 柱状图")


    if 'coverage_steps' in all_df.columns:

        cover_df = all_df.dropna(subset=['coverage_steps'])

        if not cover_df.empty:

            plt.figure(figsize=(15, 10))

            types = cover_df['type'].unique()
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for i, graph_type in enumerate(types):
                if i < len(axes):
                    ax = axes[i]
                    type_data = cover_df[cover_df['type'] == graph_type]

                    if not type_data.empty:
                        sns.barplot(data=type_data, x='graph_id', y='coverage_steps',
                                  hue='algorithm', palette=colors, ax=ax)
                        ax.set_title(f'{graph_type} - Coverage Steps')
                        ax.set_xlabel('Graph ID')
                        ax.set_ylabel('Coverage Steps (log scale)')
                        ax.set_yscale('log')
                        ax.tick_params(axis='x', rotation=45)


            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'figure5_coverage_steps_bar.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("图5已保存: Coverage Steps 柱状图")


    if 'coverage_time' in all_df.columns:

        cover_time_df = all_df.dropna(subset=['coverage_time'])

        if not cover_time_df.empty:

            plt.figure(figsize=(15, 10))

            types = cover_time_df['type'].unique()
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()

            for i, graph_type in enumerate(types):
                if i < len(axes):
                    ax = axes[i]
                    type_data = cover_time_df[cover_time_df['type'] == graph_type]

                    if not type_data.empty:
                        sns.barplot(data=type_data, x='graph_id', y='coverage_time',
                                  hue='algorithm', palette=colors, ax=ax)
                        ax.set_title(f'{graph_type} - Coverage Time')
                        ax.set_xlabel('Graph ID')
                        ax.set_ylabel('Coverage Time (log scale)')
                        ax.set_yscale('log')
                        ax.tick_params(axis='x', rotation=45)


            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'figure7_coverage_time_bar.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            print("图7已保存: Coverage Time 柱状图")

def main():
    """主函数"""
    print("开始加载数据...")

    try:

        all_df = load_and_process_data()

        if all_df.empty:
            print("错误：无法加载数据或数据为空")
            return

        print(f"成功加载数据，共 {len(all_df)} 行")
        print(f"列名: {list(all_df.columns)}")


        create_visualizations(all_df)

        print("\n所有图表已生成完成！")
        print(f"图表保存在: {os.path.join(root_dir, 'visualizations')}")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
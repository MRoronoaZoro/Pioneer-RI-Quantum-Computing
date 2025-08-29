# FILE: DTQW-NoOracle.py
# 去除Oracle算子的普通DTQW实现
import os
import json
import csv
import time
import math
import numpy as np
import pennylane as qml
import random
from collections import deque

# --- 全局配置 ---
ROOT_PATH = r'C:\Users\59415\Desktop\Pioneer RI\Research\Code'
DATASETS_PATH = os.path.join(ROOT_PATH, 'datasets')
RESULTS_PATH = os.path.join(ROOT_PATH, 'results')
ALGORITHM_NAME = 'DTQW_NoOracle'
MAX_STEPS = 100  # 增加以覆盖更多
SHOTS = 10000  # 增加以提高采样准确性
MAX_QUBITS = 15  # 限制以避免内存问题
EPSILON = 1e-10  # 概率阈值用于判断节点被"访问"

def extract_structure_params(graph_data):
    """提取结构参数"""
    graph_id = graph_data.get('graph_id', '').lower()
    structure_info = {}
    
    if 'branch' in graph_id and 'height' in graph_id:
        parts = graph_id.split('_')
        for part in parts:
            if part.startswith('branch'):
                structure_info['branch'] = int(part.replace('branch', ''))
            elif part.startswith('height'):
                structure_info['height'] = int(part.replace('height', ''))
    elif 'grid' in graph_id:
        parts = graph_id.replace('grid_', '').split('_')
        for part in parts:
            if 'x' in part:
                try:
                    dims = part.split('x')
                    structure_info['rows'] = int(dims[0])
                    structure_info['cols'] = int(dims[1])
                except:
                    pass
    elif 'hypercube' in graph_id:
        parts = graph_id.split('_')
        for part in parts:
            if part.startswith('dim'):
                structure_info['dimension'] = int(part.replace('dim', ''))
    
    return structure_info

def get_target_node(neighbors, graph_id, structure_type):
    """获取合适的目标节点"""
    num_nodes = len(neighbors)
    
    if structure_type == 'Tree':
        # 寻找叶子节点
        for node in range(num_nodes - 1, -1, -1):
            if len(neighbors[node]) == 1 and node != 0:  # 叶子节点但不是根
                return node
        return num_nodes - 1
    elif structure_type == 'Grid':
        return num_nodes - 1  # 对角最远点
    elif structure_type == 'Hypercube':
        return num_nodes - 1  # 相对顶点
    else:
        # SmallWorld, GluedTree
        return fast_farthest_node(neighbors, 0)

def fast_farthest_node(neighbors, start_node):
    """快速BFS找最远节点"""
    if not neighbors: return 0
    
    visited = {start_node}
    queue = [(start_node, 0)]
    farthest_node = start_node
    max_depth = 0
    head = 0
    
    while head < len(queue):
        node, depth = queue[head]
        head += 1
        
        if depth > max_depth:
            max_depth = depth
            farthest_node = node
            
        if depth > 8: continue
        
        for neighbor in neighbors[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    
    return farthest_node

def calculate_graph_diameter(neighbors, start_node, target_node):
    """使用BFS计算从start到target的距离（作为直径估计）"""
    if not neighbors: return 1
    
    visited = set()
    queue = deque([(start_node, 0)])
    visited.add(start_node)
    
    while queue:
        node, dist = queue.popleft()
        if node == target_node:
            return dist
        
        for neighbor in neighbors[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    return 1  # fallback if not connected

def calculate_correct_theoretical_steps(graph_data, structure_type, metric_type="hitting_time"):
    """
    基于量子随机游走理论的准确步数计算（无Oracle版本）
    """
    num_nodes = len(graph_data['neighbors'])
    structure_info = extract_structure_params(graph_data)
    
    if structure_type == 'Tree':
        if metric_type == "hitting_time":
            theoretical_steps = int(math.ceil(math.sqrt(num_nodes)))
            print(f"        Theory steps = ceil(√{num_nodes}) ≈ {theoretical_steps}")
        else:  # cover_time
            theoretical_steps = int(np.ceil(num_nodes * np.log(max(num_nodes, 2))))
        return max(theoretical_steps, 1)
    
    elif structure_type == 'Grid':
        if metric_type == "hitting_time":
            theoretical_steps = int(math.ceil(math.sqrt(num_nodes)))
            print(f"        Theory steps = ceil(√{num_nodes}) ≈ {theoretical_steps}")
        else:  # cover_time
            theoretical_steps = int(np.ceil(num_nodes * np.log(max(num_nodes, 2))))
        return max(theoretical_steps, 1)
    
    elif structure_type == 'Hypercube':
        dimension = structure_info.get('dimension', int(np.log2(num_nodes))) if num_nodes > 0 else 0
        if dimension > 0:
            if metric_type == "hitting_time":
                theoretical_steps = int(math.ceil(math.sqrt(num_nodes)))
                print(f"        Theory steps = ceil(√{num_nodes}) ≈ {theoretical_steps}")
            else:  # cover_time
                theoretical_steps = int(np.ceil(num_nodes * np.log(max(num_nodes, 2))))
            return max(theoretical_steps, 1)
        else:
            return 1
    
    elif structure_type == 'SmallWorld':
        if metric_type == "hitting_time":
            theoretical_steps = int(math.ceil(math.sqrt(num_nodes)))
            print(f"        Theory steps = ceil(√{num_nodes}) ≈ {theoretical_steps}")
        else:  # cover_time
            theoretical_steps = int(np.ceil(num_nodes * np.log(max(num_nodes, 2))))
        return max(theoretical_steps, 1)
    
    elif structure_type == 'GluedTree':
        if metric_type == "hitting_time":
            theoretical_steps = int(math.ceil(math.sqrt(num_nodes)))
            print(f"        Theory steps = ceil(√{num_nodes}) ≈ {theoretical_steps}")
        else:  # cover_time
            theoretical_steps = int(np.ceil(num_nodes))
        return max(theoretical_steps, 1)
    
    else: # Generic case
        if metric_type == "hitting_time":
            theoretical_steps = int(math.ceil(math.sqrt(num_nodes)))
            print(f"        Theory steps = ceil(√{num_nodes}) ≈ {theoretical_steps}")
        else:
            theoretical_steps = int(np.ceil(num_nodes * np.log(max(num_nodes, 2))))
        return max(theoretical_steps, 1)

def build_shift_unitary(num_nodes, neighbors, num_pos_qubits, num_coin_qubits):
    """构建 shift 操作的 unitary 置换矩阵"""
    total_qubits = num_pos_qubits + num_coin_qubits
    dim = 1 << total_qubits
    
    if total_qubits > MAX_QUBITS:
        raise ValueError(f"Too many qubits ({total_qubits} > {MAX_QUBITS}), cannot build matrix.")
    
    perm = [0] * dim
    coin_mask = (1 << num_coin_qubits) - 1
    
    for state in range(dim):
        pos = state >> num_coin_qubits
        coin = state & coin_mask
        
        if pos < num_nodes:
            neigh_list = neighbors[pos]
            deg = len(neigh_list)
            if coin < deg:
                w = neigh_list[coin]
                d_prime = next((i for i, n in enumerate(neighbors[w]) if n == pos), coin)  # 反向币标签，fallback to coin
            else:
                w = pos
                d_prime = coin  # 自循环
        else:
            w = pos
            d_prime = coin  # 无效节点自循环
        
        new_state = (w << num_coin_qubits) | d_prime
        perm[state] = new_state
    
    U = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        U[perm[i], i] = 1.0
    
    return U

def create_dtqw_circuit(
        neighbors,
        target_steps,
        start_node,
        shift_U,
        num_pos_qubits,
        num_coin_qubits,
        target_node=None,
        with_shots: bool = False,
):
    """
    构建普通DTQW电路（无Oracle算子）
    主要变化：
    1. 移除Oracle相关代码
    2. 移除Grover操作
    3. 只保留基本的量子游走演化
    """
    num_nodes = len(neighbors)
    if num_nodes == 0:
        raise ValueError("Graph has no nodes.")
    
    total_qubits = num_pos_qubits + num_coin_qubits
    pos_qubits = list(range(num_pos_qubits))
    coin_qubits = list(range(num_pos_qubits, total_qubits))
    
    dev = qml.device("lightning.qubit", wires=total_qubits, shots=SHOTS if with_shots else None)
    
    @qml.qnode(dev)
    def dtqw_circuit():
        # ========== INITIALIZATION ==========
        # 初始化位置寄存器在起始节点
        bin_start = bin(start_node)[2:].zfill(num_pos_qubits)
        for i in range(num_pos_qubits):
            if bin_start[i] == '1':
                qml.PauliX(wires=pos_qubits[i])
        
        # 初始化硬币寄存器（均匀叠加）
        for q in coin_qubits:
            qml.Hadamard(wires=q)
        
        # ========== DTQW EVOLUTION ==========
        for step in range(target_steps):
            # ----- 硬币操作：Hadamard门 -----
            for q in coin_qubits:
                qml.Hadamard(wires=q)
            
            # ----- SHIFT OPERATOR -----
            qml.QubitUnitary(shift_U, wires=range(total_qubits))
        
        # ========== MEASUREMENT ==========
        if with_shots:
            # 只测量位置量子比特
            return qml.sample(wires=pos_qubits)
        else:
            # 返回位置上的概率分布
            return qml.probs(wires=pos_qubits)
    
    return dtqw_circuit

def calculate_coverage_steps(
    neighbors,
    start_node,
    theoretical_cover_steps,
    shift_U,
    num_pos_qubits,
    num_coin_qubits,
    repetitions: int = 10,
):
    """
    运行覆盖游走（无oracle）repetitions次。
    对于每次重复，我们从1开始增加步数，直到至少90%的顶点概率>EPSILON。
    返回平均最小步数和平均运行时间。
    """
    num_nodes = len(neighbors)
    if num_nodes == 0:
        return 1, 0.0
    
    target_cnt = int(np.ceil(num_nodes * 0.9))
    steps_list = []
    time_list = []
    
    for rep in range(repetitions):
        visited = set()
        start = time.time()
        found = False
        
        for steps in range(1, MAX_STEPS + 1):
            try:
                circ = create_dtqw_circuit(
                    neighbors,
                    steps,
                    start_node,
                    shift_U,
                    num_pos_qubits,
                    num_coin_qubits,
                    target_node=None,          # 无oracle
                    with_shots=False,
                )
                prob_vec = circ()[:num_nodes]          # 截断填充
                newly = {i for i, p in enumerate(prob_vec) if p > EPSILON}
                visited.update(newly)
                
                if len(visited) >= target_cnt:
                    steps_list.append(steps)
                    found = True
                    break
            except Exception as exc:
                print(f"        Coverage step {steps} failed (rep {rep+1}): {exc}")
                continue
        
        elapsed = time.time() - start
        time_list.append(elapsed)
        
        if not found:
            steps_list.append(MAX_STEPS)
            print(
                f"        Rep {rep+1}: coverage not reached within {MAX_STEPS} steps."
            )
    
    avg_steps = float(np.mean(steps_list))
    avg_time = float(np.mean(time_list))
    # 保持真实的平均时间，不设置人为最小值
    
    print(
        f"        Coverage avg steps over {repetitions} runs: {avg_steps:.2f}"
    )
    print(
        f"        Coverage avg time  over {repetitions} runs: {avg_time:.3f}s"
    )
    
    return avg_steps, avg_time

def calculate_hitting_probability(
    neighbors,
    start_node,
    target_node,
    theoretical_steps,
    shift_U,
    num_pos_qubits,
    num_coin_qubits,
):
    """
    运行普通DTQW（无Oracle）在理论步数下的目标节点到达概率。
    使用SHOTS次采样计算概率。
    """
    try:
        t0 = time.perf_counter()
        circ = create_dtqw_circuit(
            neighbors,
            theoretical_steps,
            start_node,
            shift_U,
            num_pos_qubits,
            num_coin_qubits,
            target_node=target_node,
            with_shots=True,
        )
        samples = circ()
        runtime = time.perf_counter() - t0
        
        # -------------------------------------------------
        # 解码二进制行 → 整数节点索引
        # -------------------------------------------------
        bits = samples.astype(int)
        powers = 2 ** np.arange(num_pos_qubits - 1, -1, -1)
        node_idx = bits @ powers               # (shots,)
        
        # -------------------------------------------------
        # 丢弃"泄漏"样本（位置 >= N）
        # -------------------------------------------------
        valid_mask = node_idx < len(neighbors)
        valid_idx  = node_idx[valid_mask]
        
        # -------------------------------------------------
        # 计算有效样本上的命中概率
        # -------------------------------------------------
        if valid_idx.size == 0:
            prob = 0.0
            hits = 0
        else:
            hits = np.count_nonzero(valid_idx == target_node)
            prob = hits / valid_idx.size    # **正确的分母**
        
        print(f"Hitting probability (hits/valid): {hits}/{valid_idx.size} = {prob:.4f}")
        # 保持真实的运行时间，不设置人为最小值
        return float(prob), float(runtime)
    
    except Exception as exc:  # pragma: no cover
        print(f"Hitting probability calculation failed: {exc}")
        return 0.0, 1e-5

def run_trials_for_graph(graph_data, structure_type):
    """
    对于单个图：
        * 构建shift矩阵，
        * 计算理论命中/覆盖步数，
        * 运行覆盖10次 → 平均步数和平均时间，
        * 在理论命中步数下运行一次命中概率计算（1000次采样）。
    返回CSV写入器期望的元组：
        (coverage_steps, coverage_time, hitting_probability,
         avg_runtime, hitting_steps)
    """
    graph_id = graph_data.get("graph_id", "unknown")
    neighbors = graph_data["neighbors"]
    start_node = graph_data.get("start_node", 0)
    target_node = graph_data.get("target_node")
    
    if target_node is None:
        target_node = get_target_node(neighbors, graph_id, structure_type)
    
    num_nodes = len(neighbors)
    
    # ------------------- QUBIT COUNTS -------------------
    num_pos_qubits = max(1, (num_nodes - 1).bit_length())
    max_deg = max((len(l) for l in neighbors), default=1)
    num_coin_qubits = max(1, (max_deg - 1).bit_length())
    total_qubits = num_pos_qubits + num_coin_qubits
    
    # ------------------- FALLBACK FOR HUGE GRAPHS -------------------
    if total_qubits > MAX_QUBITS or (1 << num_pos_qubits) > 1e6:
        print(
            f"      ⚠️ Too many qubits ({total_qubits}) – using ONLY theoretical values."
        )
        theory_hit = calculate_correct_theoretical_steps(
            graph_data, structure_type, "hitting_time"
        )
        theory_cov = calculate_correct_theoretical_steps(
            graph_data, structure_type, "cover_time"
        )
        return (
            int(theory_cov),      # 无法测量覆盖 → 使用理论值
            1e-6,                 # 最小运行时间（避免0值）
            0.0,                  # 无命中执行
            1e-6,                 # 最小运行时间（避免0值）
            int(theory_hit),      # 理论命中步数
        )
    
    # ------------------- BUILD SHIFT UNITARY -------------------
    try:
        shift_U = build_shift_unitary(
            num_nodes, neighbors, num_pos_qubits, num_coin_qubits
        )
    except Exception as exc:
        print(f"      ❌ Failed to build shift matrix: {exc}")
        theory_hit = calculate_correct_theoretical_steps(
            graph_data, structure_type, "hitting_time"
        )
        theory_cov = calculate_correct_theoretical_steps(
            graph_data, structure_type, "cover_time"
        )
        return (
            int(theory_cov),
            1e-6,                 # 最小运行时间（避免0值）
            0.0,
            1e-6,                 # 最小运行时间（避免0值）
            int(theory_hit),
        )
    
    # ------------------- THEORETICAL STEP COUNTS -------------------
    theory_hit = calculate_correct_theoretical_steps(
        graph_data, structure_type, "hitting_time"
    )
    theory_cov = calculate_correct_theoretical_steps(
        graph_data, structure_type, "cover_time"
    )
    
    # ------------------- COVERAGE (10× average) -------------------
    coverage_steps, coverage_time = calculate_coverage_steps(
        neighbors,
        start_node,
        theory_cov,
        shift_U,
        num_pos_qubits,
        num_coin_qubits,
        repetitions=10,
    )
    
    # ------------------- HITTING (single run, 1000 shots) -------------------
    hitting_prob, hitting_time = calculate_hitting_probability(
        neighbors,
        start_node,
        target_node,
        theory_hit,
        shift_U,
        num_pos_qubits,
        num_coin_qubits,
    )
    
    print(f"      📊 Final Results:")
    print(
        f"          Coverage → avg {coverage_steps:.2f} steps (theory {theory_cov})"
    )
    print(
        f"          Hitting  → prob {hitting_prob:.4f} at {theory_hit} steps (time {hitting_time:.3f}s)"
    )
    
    # -------------------------------------------------------------
    # 返回CSV写入器期望顺序的值
    # -------------------------------------------------------------
    return (
        int(round(coverage_steps)),   # 平均覆盖步数（四舍五入为整数）
        float(coverage_time),         # 平均覆盖时间 [s]
        float(hitting_prob),          # 命中概率
        float(hitting_time),          # 单次命中运行的运行时间 [s]
        int(theory_hit),              # 理论命中步数
    )

def main():
    """主函数"""
    algo_results_path = os.path.join(RESULTS_PATH, f'results-{ALGORITHM_NAME}')
    os.makedirs(algo_results_path, exist_ok=True)
    
    structure_types = ['Tree', 'Grid', 'SmallWorld', 'GluedTree', 'Hypercube']
    all_results = []
    total_start_time = time.time()
    
    for struct_type in structure_types:
        print(f"\n{'=' * 80}")
        print(f"📐 Processing Structure Type: {struct_type}")
        print(f"{'=' * 80}")
        
        dataset_dir = os.path.join(DATASETS_PATH, f'datasets-{struct_type.lower()}s')
        dataset_file = os.path.join(dataset_dir, f'{struct_type.lower()}s_all_variants_quantum.json')
        
        if not os.path.exists(dataset_file):
            print(f"❌ Dataset not found for {struct_type}: {dataset_file}")
            continue
        
        try:
            with open(dataset_file, 'r') as f:
                all_graphs_data = json.load(f)
            
            if not isinstance(all_graphs_data, list):
                all_graphs_data = [all_graphs_data]
            
            print(f"📈 Found {len(all_graphs_data)} graphs for {struct_type}.")
            
            for idx, graph_data in enumerate(all_graphs_data):
                print(f"\n📊 Processing Graph {idx + 1}/{len(all_graphs_data)}: {graph_data.get('graph_id', 'N/A')}")
                
                try:
                    if 'neighbors' not in graph_data or not graph_data['neighbors']:
                        print("  Skipping graph with no neighbors.")
                        continue
                    
                    if 'start_node' not in graph_data:
                        graph_data['start_node'] = 0
                    
                    if 'target_node' not in graph_data:
                        graph_data['target_node'] = get_target_node(
                            graph_data['neighbors'], graph_data.get('graph_id', ''), struct_type
                        )
                    
                    # 运行分析
                    coverage_steps, coverage_time, hitting_probability, hitting_time, hitting_steps = run_trials_for_graph(
                        graph_data, struct_type
                    )
                    
                    # 构建结果记录
                    result = {
                        '数据名称': graph_data.get('graph_id', f'{struct_type}_{idx}'),
                        '类型': struct_type,
                        'coverage_steps': coverage_steps,
                        'coverage_time': coverage_time,
                        'hitting_probability': hitting_probability,
                        'search_time': hitting_time,
                        'hitting_steps': hitting_steps
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    print(f"  ❌ Critical error processing graph {graph_data.get('graph_id', 'N/A')}: {e}")
        
        except Exception as e:
            print(f"❌ Fatal error loading or processing dataset for {struct_type}: {e}")
    
    # 保存结果
    if all_results:
        output_file = os.path.join(algo_results_path, 'DTQW_NoOracle_Results.csv')
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['数据名称', '类型', 'coverage_steps', 'coverage_time', 'search_probability', 'search_time', 'search_steps', 'hitting_probability', 'hitting_steps'])
                writer.writeheader()
                writer.writerows(all_results)
            
            total_time = time.time() - total_start_time
            print(f"\n🎉 DTQW NO-ORACLE ANALYSIS COMPLETED!")
            print(f"⏱️  Total time: {total_time:.1f}s")
            print(f"📁 Results saved to: {output_file}")
            print(f"📊 Processed {len(all_results)} graphs total")
            
            # 显示关键统计
            if all_results:
                avg_hitting_prob = np.mean([r['hitting_probability'] for r in all_results])
                high_prob_count = sum(1 for r in all_results if r['hitting_probability'] > 0.1)
                print(f"\n📈 Key Statistics:")
                print(f"   Average Hitting Probability: {avg_hitting_prob:.4f}")
                print(f"   Graphs with >10% hitting probability: {high_prob_count}/{len(all_results)}")
        
        except Exception as e:
            print(f"❌ Error saving results to CSV: {e}")
    else:
        print(f"❌ No results generated!")

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    main()
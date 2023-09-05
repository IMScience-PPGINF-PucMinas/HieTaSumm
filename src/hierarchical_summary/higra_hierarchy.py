from typing import List, Tuple, Optional
from pathlib import Path
import networkx as nx
import higra as hg
import numpy as np
from src.hierarchical_summary.image_processing import rgb_sim_preprocess_image
from src.hierarchical_summary.math_utils import calculate_cosine_similarity, calculate_start_index, calculate_end_index
from src.infrastructure.file_utils import write_to_file, write_cut_graph_to_file


def compute_hierarchy(input_graph: nx.Graph, is_binary: bool, input_higra: Optional[Path] = None) -> \
        Tuple[List[int], hg.Tree]:
    leaf_list: List[int] = []
    graph: hg.UndirectedGraph = hg.UndirectedGraph()
    graph.add_vertices(max(input_graph.nodes()) + 1)
    edge_list: List[Tuple[int, int]] = list(input_graph.edges())

    for i in range(len(edge_list)):
        graph.add_edge(edge_list[i][0], edge_list[i][1])

    edge_weights: np.ndarray = np.empty(shape=len(edge_list))
    sources, targets = zip(*edge_list)

    for i in range(len(sources)):
        edge_weights[i] = int(input_graph[sources[i]][targets[i]]["weight"])
    nb_tree, nb_altitudes = hg.watershed_hierarchy_by_area(graph, edge_weights)

    if is_binary:
        tree, node_map = hg.tree_2_binary_tree(nb_tree)
    else:
        tree = nb_tree

    for n in tree.leaves_to_root_iterator():
        leaf: int = -2
        if tree.is_leaf(n):
            leaf = -1
            leaf_list.append(n)
        if input_higra is not None:
            write_to_file(input_higra, n, tree.parent(n), leaf)
    return leaf_list, tree


def higra_to_nx(higra_tree: hg.Tree) -> Tuple[nx.Graph, int]:
    root: int = higra_tree.root()
    parents: List[int] = higra_tree.parents()
    n: int = higra_tree.num_vertices()
    graph: nx.Graph = nx.Graph()
    for i in range(n):
        p: int = parents[i]
        if p != i:
            graph.add_edge(i, p)
    return graph, root


def gen_min_spanning_tree(input_graph: nx.Graph, output_mst_file_path: Optional[Path] = None) -> nx.Graph:
    mst: nx.Graph = nx.minimum_spanning_tree(input_graph)
    if output_mst_file_path:
        with open(str(output_mst_file_path), 'w') as output_file:
            for edge in mst.edges(data=True):
                output_file.write(f"{edge[0]}, {edge[1]}, {edge[2]['weight']}\n")

    return mst


def cut_mst_graph(graph: hg.Tree, cut_number: int, output_cut_graph_file_path: Optional[Path] = None) -> nx.Graph:
    cut_graph, root = higra_to_nx(graph)
    roots_to_remove: List[int] = [root]

    while len(roots_to_remove) < cut_number:
        new_nodes_to_remove: List[int] = []
        for root in roots_to_remove:
            for neighbor in cut_graph.neighbors(root):
                if neighbor not in roots_to_remove and neighbor not in new_nodes_to_remove \
                        and len(roots_to_remove) < cut_number - len(new_nodes_to_remove) and not \
                        graph.is_leaf(neighbor):
                    new_nodes_to_remove.append(neighbor)
        roots_to_remove.extend(new_nodes_to_remove)

    for i in roots_to_remove:
        cut_graph.remove_node(i)

    if output_cut_graph_file_path:
        write_cut_graph_to_file(output_cut_graph_file_path, cut_graph, graph)

    return cut_graph


def calculate_best_cut_number(video_file: Path, delta_t: int) -> int:
    video_path: Path = Path(video_file)
    if video_path.exists():
        frame_list: List[Path] = sorted(video_path.iterdir())
        frame_list = [frame for frame in frame_list if not frame.name.endswith(".txt")]
        features_list: List[np.ndarray] = [rgb_sim_preprocess_image(str(frame)) for frame in frame_list]
        weight_list: List[float] = []

        for vertex1 in range(len(features_list)):
            for vertex2 in range(calculate_start_index(vertex1, delta_t),
                                 calculate_end_index(vertex1, delta_t, len(features_list))):
                similarity: float = calculate_cosine_similarity(features_list[vertex1], features_list[vertex2])
                weight_list.append(similarity)

        print(video_path)
        rounded_std_deviation: int = round(np.std(weight_list) - 1)
        print(rounded_std_deviation)
        return rounded_std_deviation

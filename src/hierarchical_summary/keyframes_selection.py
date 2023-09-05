from pathlib import Path
import networkx as nx
from typing import List
from src.infrastructure.file_utils import write_to_file


def select_keyframes(graph: nx.Graph, key_frame: Path, leaf_list: List[int]) -> None:
    subgraph = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    for c in range(len(subgraph)):
        central_node = len(subgraph[c].nodes)
        comp_leaf_list = []
        for i in range(central_node):
            if list(subgraph[c])[i] in leaf_list:
                comp_leaf_list.append(list(subgraph[c])[i])
        cn = int(len(comp_leaf_list) / 2)
        kf = str(comp_leaf_list[cn]).zfill(6)
        write_to_file(key_frame, kf, '  ', '.jpg')

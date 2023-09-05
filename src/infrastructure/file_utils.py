from typing import Union, List
from pathlib import Path
import higra as hg
import networkx as nx


def write_to_file(file: Path, v1: Union[str, int], v2: Union[str, int], weight: Union[str, float]) -> None:
    f = open(file, "a")
    if v2 is None:
        data = "{}\n".format(v1)
    elif weight == ".jpg":
        data = "{}{}\n".format(v1, weight)
    else:
        data = "{}, {}, {}\n".format(v1, v2, weight)
    f.write(data)
    f.close()


def write_graph_to_file(output_graph_file: Path, graph: nx.Graph) -> None:
    with open(output_graph_file, 'w') as file:
        for edge in graph.edges(data=True):
            vertex1, vertex2, data = edge
            weight = data['weight']
            file.write(f"{vertex1}, {vertex2}, {weight:.2f}\n")


def write_cut_graph_to_file(output_cut_graph_file_path: Path, cut_graph: nx.Graph, original_graph: hg.Tree) -> None:
    with open(output_cut_graph_file_path, 'w') as output_file:
        for edge in cut_graph.edges(data=True):
            output_file.write(f"{edge[0]}, {edge[1]}, {-1 if original_graph.is_leaf(edge[0]) else -2}\n")


def load_frames_from_file(video_file: Path) -> List[str]:
    if not video_file.exists():
        raise ValueError(f"The file {video_file} does not exist.")
    return sorted(entry.name for entry in video_file.iterdir() if entry.is_file())

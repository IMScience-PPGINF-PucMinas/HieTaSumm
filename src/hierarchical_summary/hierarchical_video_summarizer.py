from typing import List, Optional, Tuple
from keras.applications import VGG16, ResNet50
from keras import Model
from pathlib import Path
import networkx as nx
import numpy as np
import torch
from .higra_hierarchy import gen_min_spanning_tree, compute_hierarchy, calculate_best_cut_number, cut_mst_graph
from .image_processing import extract_resnet_features
from .keyframes_selection import select_keyframes
from .math_utils import calculate_cosine_similarity, calculate_start_index, calculate_end_index
from ..infrastructure.file_utils import write_graph_to_file, load_frames_from_file


class HierarchicalVideoSummarizer:
    def __init__(self, model: str, rate: int, binary_hierarchy: bool = True, overwrite: bool = True):
        self.overwrite = overwrite
        self.selected_model: str = model
        self.device: torch.device = _select_torch_device()
        self.rate: int = rate
        self.available_models: List[str] = ['vgg16', 'resnet50']
        self.selected_model: str = 'resnet50'
        self.binary_hierarchy: bool = binary_hierarchy

    def get_key_frames(self, frames_dir: Path) -> None:
        frames_dir_path: Path = Path(frames_dir)
        try:
            if frames_dir_path.exists() and frames_dir_path.is_dir():
                video_list: List[Path] = sorted(frames_dir_path.iterdir())
                for video in video_list:
                    if video.is_dir():
                        if Path(video / 'keyframes.txt') and not self.overwrite:
                            print(f"Skipping video {video} as keyframes.txt already exists.")
                        else:
                            for file in video.glob('*.txt'):
                                print(f"Deleting {file}")
                                file.unlink()
                            self.get_key_frames_for_single_video(video)
            else:
                raise ValueError(f"The directory {frames_dir} does not exist or is not a valid directory.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_key_frames_for_single_video(self, video_frames_dir: Path):
        # Load frames from the video and create a graph
        graph, weight_list, features = self.build_video_graph(video_frames_dir,
                                                              video_frames_dir / 'graph.txt')

        # Generate a minimum spanning tree for the graph
        mst: nx.Graph = gen_min_spanning_tree(graph, video_frames_dir / 'mst.txt')

        # Create a hierarchy based on the minimum spanning tree and return the leaves of the new hierarchy
        hierarchy_leaves, hierarchy_tree = compute_hierarchy(mst, self.binary_hierarchy,
                                                             video_frames_dir / 'higra.txt')

        # Compute the best cut number for the hierarchy
        cut_level: int = calculate_best_cut_number(video_frames_dir, self.rate)

        # Create a new graph based on the hierarchy and the level cut
        cut_graph: nx.Graph = cut_mst_graph(hierarchy_tree, cut_level, video_frames_dir / 'cut_graph.txt')

        # Create a keyframe to represent each component or segment of the video
        select_keyframes(cut_graph, video_frames_dir / 'keyframe.txt', hierarchy_leaves)

    def build_video_graph(self, video_file: Path, output_graph_file: Optional[Path] = None) -> \
            Tuple[nx.Graph, List[float], List[np.ndarray]]:
        similarity_scores: List[float] = []
        graph: nx.Graph = nx.Graph()
        video_frames: List[str] = load_frames_from_file(video_file)
        video_frames.sort()
        video_features: List[np.ndarray] = self.extract_video_features(video_file, video_frames)

        for vertex1 in range(len(video_features)):
            for vertex2 in range(*self.get_vertex_range(vertex1, len(video_features))):
                similarity: float = calculate_cosine_similarity(video_features[vertex1], video_features[vertex2])
                similarity_scores.append(similarity)
                graph.add_edge(vertex1, vertex2, weight=similarity)

        if output_graph_file:
            write_graph_to_file(output_graph_file, graph)

        return graph, similarity_scores, video_features

    def get_vertex_range(self, vertex: int, video_features_len: int) -> Tuple[int, int]:
        start: int = calculate_start_index(vertex, self.rate)
        end: int = calculate_end_index(vertex, self.rate, video_features_len)
        return start, end

    def extract_video_features(self, video_file: Path, video_frames: List[str]) -> List[np.ndarray]:
        model = self.load_model(self.selected_model, include_top=True)
        print(f"extracting features for {video_file}")
        return [extract_resnet_features(str(video_file / frame), model) for frame in video_frames]

    def load_model(self, model_name: str, include_top: bool = True) -> Model:
        model = None
        if self.selected_model in self.available_models:
            # Load a Keras instance
            try:
                if model_name == 'vgg16':
                    model = VGG16(weights='imagenet', include_top=include_top)
                    print(f">> '{model.name}' model successfully loaded!")
                elif model_name == 'resnet50':
                    model = ResNet50(weights='imagenet', include_top=include_top)
                    print(f">> '{model.name}' model successfully loaded!")
            except (ImportError, ValueError) as e:
                print(f">> Error while loading model '{self.selected_model}': {str(e)}")

        # Wrong selected model
        else:
            print(f">> Error: there is no '{self.selected_model}' in {self.available_models}")

        return model

    def print_available_and_selected_models(self):
        print("Available Models:", self.available_models)
        print("Selected Model:", self.selected_model)


def _select_torch_device() -> torch.device:
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return device

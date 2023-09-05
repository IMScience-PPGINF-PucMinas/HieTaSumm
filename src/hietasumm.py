from pathlib import Path
from src.frame_extraction.frame_extractor import VideoFrameExtractor
from src.hierarchical_summary.hierarchical_video_summarizer import HierarchicalVideoSummarizer


class HieTaSUMM:
    def __init__(self, video_dir: str, output_frames_dir: str, saving_frames_per_second: int, model: str = "resnet50",
                 rate: int = 16, binary_hierarchy: bool = True, overwrite: bool = True):
        self.overwrite: bool = overwrite
        self.video_dir: Path = Path(video_dir)
        self.binary_hierarchy: bool = binary_hierarchy
        self.output_frames_dir: Path = Path(output_frames_dir)
        self.frame_extractor: VideoFrameExtractor = VideoFrameExtractor(saving_frames_per_second)
        self.hierarchical_video_summarizer: HierarchicalVideoSummarizer = HierarchicalVideoSummarizer(
            model, rate, binary_hierarchy, overwrite
        )

    def extract_frames(self) -> None:
        self.frame_extractor.extract_frames(self.video_dir, self.output_frames_dir)

    def get_key_frames(self) -> None:
        self.hierarchical_video_summarizer.get_key_frames(self.output_frames_dir)

    def print_available_and_selected_models(self) -> None:
        self.hierarchical_video_summarizer.print_available_and_selected_models()

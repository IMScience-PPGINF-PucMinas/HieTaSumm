from pathlib import Path
import math
import cv2


class VideoFrameExtractor:
    def __init__(self, saving_frames_per_second: float = 4):
        self.saving_frames_per_second = saving_frames_per_second

    def extract_frames(self, video_dir: Path, output_dir: Path = Path.cwd().parent / "frames") -> None:
        if Path(video_dir).exists():
            for i in Path(video_dir).iterdir():
                print("------------------------")
                print()
                self._frame_extractor(i, output_dir)
        elif Path(video_dir).is_file():
            self._frame_extractor(Path(video_dir), output_dir)
        else:
            print("The directory or file with this path does not exist.")

    def _frame_extractor(self, video_file: Path, output_dir: Path) -> None:
        # Set up the output directory
        filename = _setup_output_directory(Path(video_file), output_dir)

        # Open the video file
        cap, total_frames, frames_to_skip = self._open_video_capture(video_file)

        # Extract the frames and save the desired frames
        _extract_frames_by_interval(cap, total_frames, frames_to_skip, filename)

        # Release the video capture object
        _release_video_capture(cap)

    def _open_video_capture(self, video_file: Path) -> tuple[cv2.VideoCapture, int, int]:
        cap = cv2.VideoCapture(str(video_file))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / cap.get(cv2.CAP_PROP_FPS)
        fps = total_frames / duration_sec
        frames_to_skip = math.floor(fps / self.saving_frames_per_second)
        return cap, total_frames, frames_to_skip


def _setup_output_directory(video_file: Path, output_dir: Path) -> Path:
    filename, _ = Path(video_file).stem, Path(video_file).suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / filename
    print("-----------------------")
    print(f"Extracting frames from {filename}")

    if not filename.is_dir():
        filename.mkdir()
    return filename


def _extract_frames_by_interval(cap: cv2.VideoCapture, total_frames: int, frames_to_skip: int, filename: Path) -> None:
    frame_number = 1
    for count in range(0, total_frames, frames_to_skip):
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        is_read, frame = cap.read()
        if not is_read:
            break
        number = str(frame_number).zfill(6)
        cv2.imwrite(str(filename / f"{number}.jpg"), frame)
        frame_number += 1


def _release_video_capture(cap: cv2.VideoCapture) -> None:
    cap.release()

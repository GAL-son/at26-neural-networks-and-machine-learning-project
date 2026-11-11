"""Finding captured sequences and streaming their frames.
A sequence is ~32 GB decoded: stream it frame by frame, never collect it in a list.
"""

from __future__ import annotations
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from .exr import FRAME_FILE, load_frame

def find_scenes(captures_dir: str | Path) -> dict[str, dict[int, Path]]:
    """{capture folder name: {frame index: file}} for every sub-folder that contains frame_*.exr files."""
    
    scenes = {}
    for folder in sorted(p for p in Path(captures_dir).iterdir() if p.is_dir()):
        frames = {int(m.group(1)): file for file in folder.iterdir() if (m := FRAME_FILE.fullmatch(file.name))}
        if frames:
            scenes[folder.name] = frames
    return scenes

def missing_frames(frames: dict[int, Path], expected: int | None) -> list[int]:
    """Gaps in 0 .. max(last found, expected - 1)."""
    
    end = max(expected or 0, max(frames) + 1)
    return [index for index in range(end) if index not in frames]

def sample_indices(indices, stride: int = 1, clip: int = 1, limit: int | None = None) -> list[int]:
    """`clip` consecutive frames every `stride` frames: stride 10, clip 2 -> 0, 1, 10, 11, 20, 21, ...
    (clip 2 keeps t-1 / t pairs for temporal analyses)."""
    
    return sorted(index for index in indices if index % stride < clip)[:limit]

class SceneFrames(Dataset):
    """Frames of one scene in the given order. An unreadable file yields (index, error message) instead of a Frame."""

    def __init__(self, scene: str, frames: dict[int, Path], indices: list[int]):
        self.scene, self.frames, self.indices = scene, frames, list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = self.indices[item]
        try:
            return load_frame(self.frames[index], self.scene)
        except Exception as error:
            return index, f"{self.frames[index]}: {error}"

def _identity(item):
    return item

def iter_frames(scene: str, frames: dict[int, Path], indices: list[int], workers: int = 4, prefetch: int = 2):
    """Frames in order, decoded in `workers` processes (0 = in this process).
    On Windows the calling script needs an `if __name__ == "__main__":` guard (workers re-import it)."""
    return DataLoader(
        SceneFrames(scene, frames, indices), 
        batch_size=None, 
        num_workers=workers,
        prefetch_factor=prefetch if workers else None, 
        collate_fn=_identity
        )





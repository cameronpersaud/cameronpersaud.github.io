"""Everything the app knows about the current reconstruction, shared by all steps."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..calibration import Calibration
from ..io import SliceStack
from ..orientation import Orientation
from ..preprocess import PreprocessSettings
from ..registration import SliceTransform
from ..volume import Volume


@dataclass
class AppState:
    # Step 1 - import
    paths: list[str] = field(default_factory=list)
    max_working_size: int | None = 1024
    stack: SliceStack | None = None
    loaded_key: tuple | None = None
    orientation: Orientation = field(default_factory=Orientation)   # cutting plane etc.

    # Step 2 - calibration
    cal: Calibration | None = None

    # Step 3 - channels & clean-up
    channel_names: list[str] = field(default_factory=list)
    channel_colors: list[tuple[int, int, int]] = field(default_factory=list)
    pre: PreprocessSettings = field(default_factory=PreprocessSettings)
    proc: list | None = None
    masks: list | None = None
    pre_key: tuple | None = None
    dye_channel: int = 0                  # channel holding the red dextran dye

    # Step 4 - alignment
    align_channel: int | None = None      # None = brain channels (all except the dye)
    robust_rotation: bool = True
    transforms: list[SliceTransform] | None = None

    # Step 5 - 3D model
    vol: Volume | None = None
    vol_key: tuple | None = None
    dye_result: object | None = None      # dye.DyeResult for the current volume

    def brain_channels(self) -> list[int]:
        """Every channel except the dye (falls back to all if there is only one)."""
        n = len(self.channel_names)
        chans = [c for c in range(n) if c != self.dye_channel]
        return chans or list(range(n))

    def reset_after_load(self):
        self.cal = None
        self.proc = self.masks = None
        self.pre_key = None
        self.transforms = None
        self.vol = None
        self.vol_key = None
        self.dye_result = None

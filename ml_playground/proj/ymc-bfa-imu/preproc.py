import logging
from typing import Literal, NewType

import fire
import numpy as np

from ml_playground.repometa import DATA_ROOT
from ml_playground.util.algo import RunlengthBlock, divup, runlength
from ml_playground.util.logging import create_colored_handler

Actor = Literal["Anakamura", "ksuzuki", "master", "ueyama"]
DateStr = str
ActionID = NewType("ActionID", int)

RECORDS: list[tuple[Actor, DateStr, list[int]]] = [
    ("Anakamura", "0628", [1, 2, 3, 4, 6, 7, 8, 9, 10]),
    ("Anakamura", "0703", [1, 2, 3, 5, 6, 7, 8, 9, 11]),
    ("Anakamura", "0628", [1, 2, 3, 4, 6, 7, 8, 9, 10, 11]),
    ("ksuzuki", "0627", [1, 2, 4, 5, 6, 7, 8, 9, 10]),
    ("ksuzuki", "0704", [1, 2, 3, 4, 5, 7, 9, 10, 11, 13]),
    ("ksuzuki", "0706", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ("master", "0620", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ("ueyama", "0628", [2, 5, 6, 8, 9, 10, 11, 13]),
    ("ueyama", "0705", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
    ("ueyama", "0706", [1, 2, 3, 4, 5, 7, 8, 9, 10]),
]

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, handlers=[create_colored_handler()])


def _generate_from_orig_imu() -> None:
    orig_root = DATA_ROOT / "ymc-bfa-imu-orig"
    dest_root = DATA_ROOT / "ymc-bfa-imu"

    i = 0
    for actor, date, try_ids in RECORDS:
        for try_id in try_ids:
            _log.info("[i=%03d] Loading %s_%s_%03d", i, actor, date, try_id)

            with open(orig_root / "groundTruth" / f"{i}.txt") as f:
                framewise_gtruth = [ActionID(int(line)) for line in f]

            longest_segs: dict[ActionID, RunlengthBlock[ActionID]] = dict()
            for seg in runlength(framewise_gtruth):
                action_id = seg.val
                if action_id not in longest_segs:
                    longest_segs[action_id] = seg
                elif seg.len > longest_segs[action_id].len:
                    longest_segs[action_id] = seg

            data = np.load(orig_root / "features" / f"{i}.npy")
            num_total_frames = len(framewise_gtruth)
            assert type(data) is np.ndarray
            assert data.shape == (18, num_total_frames)
            data = data.T

            for action_id, seg in longest_segs.items():
                path = dest_root / str(action_id) / actor / f"{date}_{try_id:03}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                d = data[seg.begin : seg.end : 3, :]  # 元データの90fpsを30fpsへリサンプリング
                assert d.shape == (
                    divup(seg.len, 3),
                    18,
                ), f"Invalid shape: {d.shape} ({seg=} {i=})"
                np.save(path, d)

            i += 1


if __name__ == "__main__":
    fire.Fire(_generate_from_orig_imu)

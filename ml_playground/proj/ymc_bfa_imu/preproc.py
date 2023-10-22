import logging

import fire
import numpy as np

from ml_playground.proj.ymc_bfa_imu.data import RECORDS, ActionID
from ml_playground.repometa import DATA_ROOT
from ml_playground.util.algo import RunlengthBlock, divup, runlength
from ml_playground.util.logging import create_colored_handler

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

"""
Flatten AnyV2V's nested results into a flat output dir matching the rest of
our baselines: outputs/baseline_anyv2v_inference_160/{id}.mp4
"""
import shutil
from pathlib import Path

WORK = Path("/home/xinghao/DiffSynth-Studio-TextVACE/outputs/anyv2v_work/results")
OUT = Path("/home/xinghao/DiffSynth-Studio-TextVACE/outputs/baseline_anyv2v_inference_160")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n_copied, n_missing = 0, []
    # Layout: {WORK}/{video_id}/{edited_video_name}/<some.mp4>
    # Since we set edited_video_name = video_id, expect {WORK}/{id}/{id}/something.mp4
    for vid_dir in sorted(WORK.iterdir()):
        if not vid_dir.is_dir():
            continue
        rid = vid_dir.name
        # find first .mp4 anywhere under this id
        mp4s = list(vid_dir.rglob("*.mp4"))
        if not mp4s:
            n_missing.append(rid)
            continue
        # If multiple, prefer the one without "recon" or with "edit" in name
        mp4s.sort(key=lambda p: ("recon" in p.name.lower(), p.name))
        src = mp4s[0]
        dst = OUT / f"{rid}.mp4"
        shutil.copy2(src, dst)
        n_copied += 1
    print(f"copied {n_copied} videos -> {OUT}")
    if n_missing:
        print(f"missing for {len(n_missing)} ids: {n_missing[:10]}{'...' if len(n_missing)>10 else ''}")


if __name__ == "__main__":
    main()

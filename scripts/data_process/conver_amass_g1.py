#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import argparse
import numpy as np
import joblib
from tqdm import tqdm

def load_qpos_npy(path: str, expected_min_cols=36) -> np.ndarray:
    """
    读取一段 qpos 的 .npy
    约定：至少包含前 36 列(7 自由根 + 29 关节）
    """
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] < expected_min_cols:
        raise ValueError(f"{path}: 需要二维数组且列数>= {expected_min_cols}，当前形状 {arr.shape}")
    return arr

def get_motion(arr:np.ndarray, add_pelvis_z: float,model_dof:int, fps:int):
    arr[:,0:3] += float(add_pelvis_z)
    root_trans_offset = arr[:,0:3].astype(np.float32).copy()
    root_rot_xyzw = arr[:,3:7].astype(np.float32).copy()
    dof = arr[:,7:7+model_dof].astype(np.float32).copy()
    motion = {
        "root_trans_offset": root_trans_offset,
        "root_rot": root_rot_xyzw,
        "dof": dof,
        "fps": int(fps)
    }
    return motion
def slice_to_motion(arr: np.ndarray, add_pelvis_z: float, model_dof: int, start: int, end: int, fps: int):
    """
    从 qpos 数组切出 [start, end) 片段并转为统一 motion dict
    """
    seg = arr[start:end]
    root_trans_offset = seg[:, 0:3].astype(np.float32).copy()
    if add_pelvis_z != 0.0:
        root_trans_offset[:, 2] += float(add_pelvis_z)
    root_rot_xyzw = seg[:, 3:7].astype(np.float32).copy()
    dof = seg[:, 7:7+model_dof].astype(np.float32).copy()

    motion = {
        "root_trans_offset": root_trans_offset,
        "root_rot": root_rot_xyzw,     # xyzw（你的可视化里会转 wxyz）
        "dof": dof,                    # (T, 29)
        "fps": int(fps),               # 仅为播放器需要；不从文件名解析
        "frame_range": [int(start), int(end)],
    }
    return motion

def convert_one_file_to_clips(path: str,
                              add_pelvis_z: float,
                              model_dof: int,
                              fps: int,
                              clip_len_frames: int,
                              clip_stride_frames: int,
                              min_clip_frames: int,
                              drop_last: bool,
                              cut:bool):
    """
    把单个 .npy 切成多个 clip,返回 {key_suffix: motion_dict}
    key_suffix 类似 '#clip000'、'#clip001'...
    仅在该文件内部切片，不跨文件。
    """
    arr = load_qpos_npy(path, expected_min_cols=7+model_dof)
    T = arr.shape[0]
    if cut == False:
        motion = get_motion(arr, add_pelvis_z, model_dof, fps)
        motion["clip_idx"] = 0
        return {f"#clip000": motion}
    
    # 不切分：整个序列作为一个 clip
    if clip_len_frames <= 0:
        motion = slice_to_motion(arr, add_pelvis_z, model_dof, 0, T, fps)
        motion["clip_idx"] = 0
        return {"#clip000": motion}

    clip_len = int(clip_len_frames)
    stride = int(clip_stride_frames if clip_stride_frames > 0 else clip_len)
    min_keep = max(1, int(min_clip_frames))

    out = {}
    clip_idx = 0
    t = 0
    while t < T:
        end = t + clip_len
        if end > T:
            if drop_last:
                break
            end = T
        if end - t >= min_keep:
            motion = slice_to_motion(arr, add_pelvis_z, model_dof, t, end, fps)
            motion["clip_idx"] = int(clip_idx)
            out[f"#clip{clip_idx:03d}"] = motion
            clip_idx += 1
        t += stride

    # 若没有任何片段满足长度，至少给一个整段
    if not out:
        motion = slice_to_motion(arr, add_pelvis_z, model_dof, 0, T, fps)
        motion["clip_idx"] = 0
        out["#clip000"] = motion

    return out

def main():
    parser = argparse.ArgumentParser("Convert qpos .npy (G1 7+29) to multi-clip .pkl (frame-based)")
    parser.add_argument("--in_dir", type=str, required=True, help="包含 *.npy(每帧 qpos)的根目录")
    parser.add_argument("--fps", type=int, default=30, help="统一写入到输出的 fps(仅为播放器使用)")
    parser.add_argument("--add_pelvis_z", type=float, default=0.0, help="给 root z 统一加偏置（如 0.793)")

    # 纯“按帧”切分
    parser.add_argument("--clip_len_frames", type=int, default=500, help="每个 clip 的帧数；<=0 表示不切分")
    parser.add_argument("--clip_stride_frames", type=int, default=500, help="相邻 clip 起点步长（帧）；<=0 则与 clip_len 相同")
    parser.add_argument("--min_clip_frames", type=int, default=15, help="最短保留的 clip 帧数，小于此阈值不保留")
    parser.add_argument("--drop_last", action="store_true", help="若末段长度不足 clip_len 则丢弃（否则保留短片段）")

    # 模型 DOF（G1=29）
    parser.add_argument("--model_dof", type=int, default=29)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "**", "*.npy"), recursive=True))
    if not files:
        print(f"[WARN] 未在 {args.in_dir} 下找到 .npy 文件")
    out_dict = {}
    skipped_files = 0
    total_clips = 0
    fps = 30
    for f in tqdm(files):
        try:
            clip_dict = convert_one_file_to_clips(
                path=f,
                add_pelvis_z=args.add_pelvis_z,
                model_dof=args.model_dof,
                fps=fps,  # 不依赖文件名
                clip_len_frames=args.clip_len_frames,
                clip_stride_frames=args.clip_stride_frames,
                min_clip_frames=args.min_clip_frames,
                drop_last=args.drop_last,
                cut=False,
            )

            parts = f.replace("\\", "/").split("/")
            tail3 = parts[-3:] if len(parts) >= 3 else parts
            key_base = "0-" + "_".join(tail3).replace(".npy", "")

            for clip_suffix, motion in clip_dict.items():
                out_key = f"{key_base}{clip_suffix}"
                out_dict[out_key] = motion
                out_dict['fps'] = fps

                total_clips += 1

        except Exception as e:
            skipped_files += 1
            print(f"[skip] {f} -> {e}")
    import ipdb; ipdb.set_trace()
    joblib.dump(out_dict, "data/amass/cmu_g1.pkl", compress=True)
    print(f"Saved {len(out_dict)} clips to {'data/amass/cmu_g1.pkl'}. "
          f"From files: {len(files)}, skipped files: {skipped_files}, total clips: {total_clips}")

if __name__ == "__main__":
    main()

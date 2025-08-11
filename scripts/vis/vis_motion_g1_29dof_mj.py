#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np
import joblib
import mujoco
import mujoco.viewer

def load_all_clips(pkl_path: str):
    """读取 pkl（dict of clips）。返回 keys(list) 和 motions(dict)。"""
    motions = joblib.load(pkl_path)
    if not isinstance(motions, dict) or len(motions) == 0:
        raise ValueError("pkl 内容为空或不是 {key: motion} 字典")
    # 仅保留包含必要字段的条目
    usable = {}
    for k, m in motions.items():
        if not isinstance(m, dict):
            continue
        if all(x in m for x in ("root_trans_offset", "root_rot", "dof")):
            usable[k] = m
    if not usable:
        raise ValueError("pkl 中没有包含 root_trans_offset/root_rot/dof 的条目")
    keys = sorted(usable.keys())
    return keys, usable

def pick_key(keys, start_key: str = "", start_idx: int = 0):
    """按优先级选择初始 clip：精确匹配start_key->包含start_key->start_idx。"""
    if start_key and start_key in keys:
        return keys.index(start_key)
    if start_key:
        for i, k in enumerate(keys):
            if start_key in k:
                return i
    start_idx = max(0, min(len(keys)-1, int(start_idx)))
    return start_idx

def clamp_int(x, lo, hi):  # inclusive clamp
    return max(lo, min(hi, int(x)))

def main():
    ap = argparse.ArgumentParser("G1-29DoF Motion Viewer (MuJoCo)")
    ap.add_argument("--model_xml", type=str,
                    default="/home/cxd/data/Retargeted_AMASS_for_robotics/robots/g1/g1_29dof_rev_1_0.xml",
                    help="G1 29DoF 的 MuJoCo 模型 xml 路径")
    ap.add_argument("--motion_pkl", type=str, required=True, help="conver_amass_g1.py 生成的 pkl（含多个 #clipXXX）")
    ap.add_argument("--egl", action="store_true", help="无显示环境渲染（MUJOCO_GL=egl）")
    ap.add_argument("--cam_dist", type=float, default=3.0)
    ap.add_argument("--cam_azimuth", type=float, default=180.0)
    ap.add_argument("--cam_elevation", type=float, default=-30.0)

    # clip 选择
    ap.add_argument("--start_key", type=str, default="", help="初始选择的 key（精确或子串匹配）")
    ap.add_argument("--start_idx", type=int, default=0, help="初始选择第几个 clip（从0开始）")

    # 播放控制
    ap.add_argument("--fps_override", type=float, default=0.0, help=">0 则覆盖 clip 内的 fps")
    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0, help="初始倍速（影响每步前进的帧数）")

    args = ap.parse_args()

    if args.egl:
        os.environ["MUJOCO_GL"] = "egl"

    # 加载模型并确认 DOF
    m = mujoco.MjModel.from_xml_path(args.model_xml)
    expected_dof = m.nq - 7
    if expected_dof <= 0:
        raise RuntimeError(f"模型 nq={m.nq}，看起来不是 free-root humanoid？")
    print(f"[INFO] Model nq={m.nq}, expected DOF (nq-7)={expected_dof}")

    # 读取 pkl 中的所有 clip
    keys, motions = load_all_clips(args.motion_pkl)
    cur_idx = pick_key(keys, args.start_key, args.start_idx)

    # 状态
    paused = False
    speed = max(0.1, float(args.speed))
    frame_jump = 1  # 可用 J 在 1 与 10 间切换
    step = max(0, int(args.start_frame))

    # 创建数据缓存（当前 clip）
    def load_clip(i):
        nonlocal step
        i = clamp_int(i, 0, len(keys)-1)
        key = keys[i]
        mot = motions[key]
        root_trans = np.asarray(mot["root_trans_offset"], dtype=np.float32)
        root_rot   = np.asarray(mot["root_rot"],          dtype=np.float32)  # xyzw
        dof        = np.asarray(mot["dof"],               dtype=np.float32)

        # 基本形状检查/对齐
        T = min(len(root_trans), len(root_rot), len(dof))
        root_trans = root_trans[:T]
        root_rot   = root_rot[:T]
        dof        = dof[:T]

        if dof.shape[1] != expected_dof:
            raise ValueError(f"{key}: dof 维度={dof.shape[1]} 与模型要求={expected_dof} 不一致")

        # fps 处理
        fps = float(args.fps_override) if args.fps_override > 0 else float(mot.get("fps", 30))
        m.opt.timestep = 1.0 / fps

        # 起始帧
        step = clamp_int(step, 0, T-1)

        # 打印信息
        fr = mot.get("frame_range", [0, T])
        clip_i = mot.get("clip_idx", None)
        extra = f" frame_range={fr}" if fr is not None else ""
        if clip_i is not None:
            extra = f" clip_idx={clip_i};" + extra
        print(f"[CLIP] {i+1}/{len(keys)} | key={key} | T={T} | fps={fps}{extra}")

        return key, root_trans, root_rot, dof, T, fps

    key, root_trans, root_rot, dof, T, fps = load_clip(cur_idx)
    d = mujoco.MjData(m)

    # 键盘回调
    def key_callback(keycode):
        nonlocal paused, speed, frame_jump, step, cur_idx, key, root_trans, root_rot, dof, T, fps
        c = ""
        try:
            c = chr(keycode) if 0 <= keycode < 256 else ""
        except Exception:
            pass

        if c == " ":
            paused = not paused
            print(f"[KEY] pause={paused}")

        elif c.upper() == "R":
            step = 0
            print("[KEY] reset to frame 0")

        elif c.upper() == "L":
            speed *= 1.5
            print(f"[KEY] speed={speed:.2f}x")

        elif c.upper() == "K":
            speed = max(0.1, speed / 1.5)
            print(f"[KEY] speed={speed:.2f}x")

        elif c.upper() == "J":
            frame_jump = 10 if frame_jump == 1 else 1
            print(f"[KEY] frame_jump={frame_jump}")

        elif keycode == 262:  # Right
            step = min(T - 1, step + frame_jump)
        elif keycode == 263:  # Left
            step = max(0, step - frame_jump)

        elif c.upper() == "N":  # 下一段
            cur_idx = clamp_int(cur_idx + 1, 0, len(keys) - 1)
            step = 0
            key, root_trans, root_rot, dof, T, fps = load_clip(cur_idx)
        elif c.upper() == "B":  # 上一段
            cur_idx = clamp_int(cur_idx - 1, 0, len(keys) - 1)
            step = 0
            key, root_trans, root_rot, dof, T, fps = load_clip(cur_idx)

        elif c.upper() == "I":  # 打印当前信息
            print(f"[INFO] key={key} | step={step}/{T-1} | speed={speed:.2f} | fps={1.0/m.opt.timestep:.1f}")

    print("Controls: Space=Pause  R=Reset  L/K=Faster/Slower  J=Step(1/10)  ←/→=Frame  N/B=Next/Prev clip  I=Info")

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        # 相机
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.7], dtype=np.float32)
        viewer.cam.distance = float(args.cam_dist)
        viewer.cam.azimuth = float(args.cam_azimuth)
        viewer.cam.elevation = float(args.cam_elevation)

        while viewer.is_running():
            t0 = time.time()

            # 写入 qpos（free root + 29 dof）
            d.qpos[:3] = root_trans[step]
            q_xyzw = root_rot[step]
            d.qpos[3:7] = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)  # xyzw -> wxyz
            d.qpos[7:7+expected_dof] = dof[step]

            mujoco.mj_forward(m, d)
            viewer.sync()

            # 推进帧
            if not paused:
                inc = max(1, int(round(speed)))
                step += inc
                if step >= T:
                    step = 0  # 循环

            # 实时控制
            remain = m.opt.timestep - (time.time() - t0)
            if remain > 0:
                time.sleep(remain)

if __name__ == "__main__":
    main()

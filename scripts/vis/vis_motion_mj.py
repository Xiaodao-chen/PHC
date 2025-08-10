#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from isaacgym import gymapi, gymtorch  
import sys
import time
import argparse
import numpy as np
import torch
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as sRot

# 保持你的工程依赖路径
sys.path.append(os.getcwd())
from easydict import EasyDict
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.motion_lib_base import FixHeightMode
from phc.utils.motion_lib_smpl import MotionLibSMPL
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot


def add_visual_capsule(scene, point1, point2, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        float(point1[0]), float(point1[1]), float(point1[2]),
        float(point2[0]), float(point2[1]), float(point2[2])
    )


def build_smpl_xml(tmp_xml="/tmp/smpl/test_good.xml"):
    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": False,
        "upright_start": True,
        "remove_toe": False,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "model": "smpl",
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }
    smpl_robot = SMPL_Robot(robot_cfg, data_dir="data/smpl")
    gender_beta = np.zeros((17))
    smpl_robot.load_from_skeleton(
        betas=torch.from_numpy(gender_beta[None, 1:]),
        gender=gender_beta[0:1],
        objs_info=None,
    )
    os.makedirs(os.path.dirname(tmp_xml), exist_ok=True)
    smpl_robot.write_xml(tmp_xml)
    return tmp_xml


def parse_args():
    p = argparse.ArgumentParser("Stable SMPL motion viewer (MuJoCo)")
    p.add_argument("--motion_file", type=str, required=True,
                   help="pkl 路径（如 data/amass/...upright.pkl 或你的 dance_sample_g1.pkl）")
    p.add_argument("--egl", action="store_true", help="在无显示/远程环境下启用 EGL (MUJOCO_GL=egl)")
    p.add_argument("--start_idx", type=int, default=0, help="起始 motion 索引（batch 加载起点）")
    p.add_argument("--num_motions", type=int, default=1, help="每次加载的 motion 数（默认 1）")
    p.add_argument("--fps", type=float, default=30.0, help="回放帧率（如 pkl 未带 fps，则使用该值")
    p.add_argument("--cam_dist", type=float, default=3.0)
    p.add_argument("--cam_azimuth", type=float, default=180.0)
    p.add_argument("--cam_elevation", type=float, default=-30.0)
    return p.parse_args()


def main():
    args = parse_args()

    if args.egl:
        os.environ["MUJOCO_GL"] = "egl"

    DEVICE = torch.device("cpu")

    # 状态变量
    curr_start = int(args.start_idx)
    num_motions = int(args.num_motions)
    motion_id = 0
    time_step = 0.0
    paused = False
    request_next = {"flag": False}

    # 生成 SMPL xml 并构造 skeleton
    xml_path = build_smpl_xml("/tmp/smpl/test_good.xml")
    sk_tree = SkeletonTree.from_mjcf(xml_path)

    # 加载 motion 库
    motion_lib_cfg = EasyDict(
        dict(
            motion_file=args.motion_file,
            device=DEVICE,
            fix_height=FixHeightMode.full_fix,
            min_length=-1,
            max_length=-1,
            im_eval=False,
            multi_thread=False,
            smpl_type="smpl",
            randomrize_heading=True
        )
    )
    motion_lib = MotionLibSMPL(motion_lib_cfg)

    def load_batch(start_idx):
        print(f"[INFO] Loading motions: start_idx={start_idx}, num={num_motions}")
        motion_lib.load_motions(
            skeleton_trees=[sk_tree] * num_motions,
            gender_betas=[torch.zeros(17)] * num_motions,
            limb_weights=[np.zeros(10)] * num_motions,
            random_sample=False,
            start_idx=start_idx,
        )
        print("[INFO] Motions loaded.")

    load_batch(curr_start)

    # 创建 MuJoCo 模型
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    dt = 1.0 / float(args.fps)
    mj_model.opt.timestep = dt

    # 检查 DOF 维度是否匹配（第一次取样）
    sample_len = motion_lib.get_motion_length(motion_id).item()
    sample_res = motion_lib.get_motion_state(torch.tensor([motion_id]), torch.tensor([0.0]))
    dof_pos_axis_angle = sample_res["dof_pos"][0].cpu().numpy().reshape(-1, 3)  # (J,3)
    dof_euler_len = sRot.from_rotvec(dof_pos_axis_angle).as_euler("XYZ").flatten().shape[0]
    expected = mj_model.nq - 7
    if dof_euler_len != expected:
        raise RuntimeError(
            f"[ERROR] DOF 长度不匹配: euler_len={dof_euler_len}, 但模型需要 {expected} (njoint DOF = nq-7). "
            f"请检查 pkl 关节数与 xml 关节数是否一致。"
        )
    print(f"[CHECK] DOF OK: model expects {expected}, motion provides {dof_euler_len}")

    # 键盘回调（轻操作：只置标志或切换暂停）
    def key_callback(keycode):
        nonlocal time_step, paused
        try:
            c = chr(keycode) if 0 <= keycode < 256 else ""
        except Exception:
            c = ""
        if c == "T":
            request_next["flag"] = True
            print("[KEY] Next motion (queued)")
        elif c == "R":
            time_step = 0.0
            print("[KEY] Reset t=0")
        elif c == " ":
            paused = not paused
            print(f"[KEY] Pause={paused}")

    # 渲染
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        # 相机
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.7])
        viewer.cam.distance = float(args.cam_dist)
        viewer.cam.azimuth = float(args.cam_azimuth)
        viewer.cam.elevation = float(args.cam_elevation)

        # 放一些红色小胶囊（可把 i 的上限换成 sk_tree 节点数）
        for _ in range(len(sk_tree._node_indices)):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.01, np.array([1, 0, 0, 1]))

        while viewer.is_running():
            loop_start = time.time()

            # 非阻塞地处理“下一段”
            if request_next["flag"]:
                try:
                    curr_start += num_motions
                    load_batch(curr_start)
                    time_step = 0.0
                except Exception as e:
                    print(f"[WARN] load next failed: {e}, rewind to start")
                    curr_start = 0
                    load_batch(curr_start)
                    time_step = 0.0
                finally:
                    request_next["flag"] = False

            # 计算当前时间戳
            motion_len = motion_lib.get_motion_length(motion_id).item()
            motion_time = time_step % motion_len

            # 取状态
            motion_res = motion_lib.get_motion_state(
                torch.tensor([motion_id]),
                torch.tensor([motion_time]),
            )

            root_pos = motion_res["root_pos"][0].cpu().numpy()          # (3,)
            root_rot = motion_res["root_rot"][0].cpu().numpy()          # (4,) xyzw
            dof_pos = motion_res["dof_pos"][0].cpu().numpy()            # (J*3,) axis-angle

            # 写入 qpos：自由基 + 关节
            mj_data.qpos[:3] = root_pos
            mj_data.qpos[3:7] = root_rot[[3, 0, 1, 2]]                  # xyzw -> wxyz
            mj_data.qpos[7:] = sRot.from_rotvec(dof_pos.reshape(-1, 3)).as_euler("XYZ").flatten()

            mujoco.mj_forward(mj_model, mj_data)

            if not paused:
                time_step += dt

            # 可视化刚体中心（rg_pos），有就画一下
            rb_pos = motion_res.get("rg_pos", None)
            if rb_pos is not None:
                rb_pos = rb_pos[0]
                n = min(rb_pos.shape[0], viewer.user_scn.ngeom)
                for i in range(n):
                    viewer.user_scn.geoms[i].pos = rb_pos[i]

            viewer.sync()

            # 控制实时性
            remain = mj_model.opt.timestep - (time.time() - loop_start)
            if remain > 0:
                time.sleep(remain)


if __name__ == "__main__":
    main()

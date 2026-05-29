#!/usr/bin/env python3
"""
单臂轨迹复现（TXT 数据源，Startouch 版本）
======================================

参考 xArm 版本 replay 脚本，保持相同的数据读取与时间调度逻辑，
控制端改为 Startouch 的 _raw 透传接口：
- 位姿：SingleArm.set_end_effector_pose_euler_raw
- 夹爪：SingleArm.setGripperPosition_raw
"""

import argparse
import os
import time
import re
import numpy as np
from scipy.spatial.transform import Rotation

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STARTOUCH_INTERFACE_DIR = os.path.join(SCRIPT_DIR, "startouch_sdk", "interface_py")
if STARTOUCH_INTERFACE_DIR not in sys.path:
    sys.path.append(STARTOUCH_INTERFACE_DIR)

from startouchclass import SingleArm, quaternion_to_euler_xyzw


def parse_time_suffix(name, prefix):
    """从类似 prefix_20260510 / prefix_164449 中提取数值时间键。"""
    if not name.startswith(f"{prefix}_"):
        return None
    suffix = name[len(prefix) + 1:]
    return int(suffix) if suffix.isdigit() else None


def parse_task_time_key(name):
    """从 task_20260508Z... 这类名称中提取首个 8 位日期作为排序键。"""
    if not name.startswith("task_"):
        return None
    m = re.search(r"(\d{8})", name)
    return int(m.group(1)) if m else None


def list_dirs_sorted(parent_dir, name_prefix):
    """列出 parent_dir 下符合前缀的目录，按时间倒序（最新在前）。"""
    if not os.path.isdir(parent_dir):
        return []

    entries = []
    for name in os.listdir(parent_dir):
        full_path = os.path.join(parent_dir, name)
        if not os.path.isdir(full_path):
            continue
        if not name.startswith(f"{name_prefix}_"):
            continue

        time_key = parse_time_suffix(name, name_prefix)
        mtime = os.path.getmtime(full_path)
        # 先按可解析时间键，再按 mtime，最后按名称
        entries.append((full_path, name, time_key if time_key is not None else -1, mtime))

    entries.sort(key=lambda x: (x[2], x[3], x[1]), reverse=True)
    return entries


def choose_one(entries, title):
    """交互选择一个目录，直接回车默认第一个（最新）。"""
    if not entries:
        raise RuntimeError(f"{title} 为空，无法选择。")

    print(f"\n请选择 {title}（回车默认 1，最新项）：")
    for idx, (_, name, _, mtime) in enumerate(entries, 1):
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        print(f"  {idx:2d}. {name:<40s} mtime={ts_str}")

    while True:
        user_in = input("> ").strip()
        if user_in == "":
            return entries[0][0], entries[0][1]
        if user_in.isdigit():
            choice = int(user_in)
            if 1 <= choice <= len(entries):
                return entries[choice - 1][0], entries[choice - 1][1]
        print(f"输入无效，请输入 1~{len(entries)}，或直接回车。")


def collect_multi_candidates(task_dir):
    """
    在 task 目录内查找 multi_session_*。
    优先 task 直接子目录，其次递归一层常见中间层（如 background）。
    """
    direct = list_dirs_sorted(task_dir, "multi_session")
    if direct:
        return direct

    candidates = []
    for sub_name in os.listdir(task_dir):
        sub_dir = os.path.join(task_dir, sub_name)
        if not os.path.isdir(sub_dir):
            continue
        candidates.extend(list_dirs_sorted(sub_dir, "multi_session"))

    # 去重并按同样规则排序
    uniq = {}
    for item in candidates:
        uniq[item[0]] = item
    merged = list(uniq.values())
    merged.sort(key=lambda x: (x[2], x[3], x[1]), reverse=True)
    return merged


def auto_select_data_paths(data_root):
    """
    交互选择 task -> multi_session -> session，返回轨迹和夹爪文件路径。
    """
    root = os.path.expanduser(data_root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"数据根目录不存在: {root}")

    tasks = []
    for name in os.listdir(root):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        if not name.startswith("task_"):
            continue
        tasks.append((full, name, parse_task_time_key(name), os.path.getmtime(full)))
    tasks.sort(key=lambda x: (x[2] if x[2] is not None else -1, x[3], x[1]), reverse=True)

    if not tasks:
        raise RuntimeError(f"在 {root} 下未找到 task_* 目录。")
    task_entries = [(p, n, tk if tk is not None else -1, mt) for (p, n, tk, mt) in tasks]
    task_dir, task_name = choose_one(task_entries, "task")

    multi_entries = collect_multi_candidates(task_dir)
    if not multi_entries:
        raise RuntimeError(f"在 {task_dir} 下未找到 multi_session_* 目录。")
    multi_dir, multi_name = choose_one(multi_entries, "multi_session")

    session_entries = list_dirs_sorted(multi_dir, "session")
    if not session_entries:
        raise RuntimeError(f"在 {multi_dir} 下未找到 session_* 目录。")
    session_dir, session_name = choose_one(session_entries, "session")

    traj_path = os.path.join(session_dir, "Merged_Trajectory", "merged_trajectory.txt")
    clamp_path = os.path.join(session_dir, "Clamp_Data", "clamp_data_tum.txt")

    print("\n已选择数据目录：")
    print(f"  task  : {task_name}")
    print(f"  multi : {multi_name}")
    print(f"  session: {session_name}")
    print(f"  traj  : {traj_path}")
    print(f"  clamp : {clamp_path}")
    return traj_path, clamp_path


def load_trajectory(txt_path):
    """
    加载 merged_trajectory.txt。
    返回: timestamps (N,), poses (N, 7) = [x_m, y_m, z_m, qx, qy, qz, qw]
    """
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"轨迹文件不存在: {txt_path}")
    data = np.loadtxt(txt_path)
    timestamps = data[:, 0]
    poses = data[:, 1:]
    return timestamps, poses


def load_clamp(clamp_path):
    """
    加载 clamp_data_tum.txt。
    返回: timestamps (M,), values (M,)
    """
    if not os.path.isfile(clamp_path):
        return None, None
    data = np.loadtxt(clamp_path)
    return data[:, 0], data[:, 1]


def interpolate_clamp(traj_ts, clamp_ts, clamp_vals):
    """将夹爪时间序列插值到轨迹时间戳上。"""
    return np.interp(traj_ts, clamp_ts, clamp_vals)


def normalize_clamp_values(values, mode="auto", eps=1e-6):
    """
    将夹爪值归一化到 [0, 1]。
    mode:
    - auto: 当值域明显不是 [0,1] 时使用 min-max 归一化，否则不变
    - minmax: 强制使用 min-max 归一化
    - none: 不归一化
    返回: (normalized_values, changed, info_text)
    """
    vals = np.asarray(values, dtype=float)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))

    use_minmax = False
    if mode == "minmax":
        use_minmax = True
    elif mode == "auto":
        # 典型非标准夹爪值（如 0.6~89）触发归一化
        if vmin < -eps or vmax > 1.0 + eps:
            use_minmax = True
    elif mode == "none":
        use_minmax = False
    else:
        raise ValueError(f"未知 clamp_norm_mode: {mode}")

    if not use_minmax:
        return vals, False, f"mode={mode}, passthrough range=[{vmin:.6f}, {vmax:.6f}]"

    denom = vmax - vmin
    if denom < eps:
        # 常量序列，退化到全 0，避免除 0
        out = np.zeros_like(vals)
        return out, True, f"mode={mode}, degenerate range=[{vmin:.6f}, {vmax:.6f}] -> zeros"

    out = (vals - vmin) / denom
    return out, True, (
        f"mode={mode}, minmax range=[{vmin:.6f}, {vmax:.6f}] "
        "-> normalized to [0,1]"
    )


def build_T_base(base_x, base_y, base_z, base_roll_deg, base_pitch_deg, base_yaw_deg):
    """
    构建基座变换矩阵 T_base（4x4）
    - 位置单位: 米
    - 角度单位: 度（xyz 欧拉）
    """
    rot = Rotation.from_euler(
        "xyz",
        [base_roll_deg, base_pitch_deg, base_yaw_deg],
        degrees=True,
    ).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [base_x, base_y, base_z]
    return T


def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base):
    """
    将局部位姿变换到机械臂 base 坐标系：
    - p_base = base_origin + p_local
    - R_base = R_local @ R_base_frame
    返回:
    - p_base: np.ndarray(3,)
    - q_base_xyzw: np.ndarray(4,)
    """
    R_local = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    R_base_frame = T_base[:3, :3]
    base_origin = T_base[:3, 3]

    R_base = R_local @ R_base_frame
    p_base = base_origin + np.array([x, y, z], dtype=float)
    q_base_xyzw = Rotation.from_matrix(R_base).as_quat()
    return p_base, q_base_xyzw


def pose7_to_startouch(pose7, T_base):
    """
    输入 pose7: [x, y, z, qx, qy, qz, qw]
    转换为 Startouch euler raw 所需格式
    返回: (pos_xyz, euler_rpy)
    """
    x, y, z, qx, qy, qz, qw = pose7
    p_base, q_base_xyzw = transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base)
    pos = p_base.tolist()
    euler_rpy = quaternion_to_euler_xyzw(q_base_xyzw)
    return pos, euler_rpy.tolist()


def clamp_to_startouch(clamp_val):
    """
    将传感器夹爪值映射到 Startouch 夹爪透传值 [0, 1]
    - setGripperPosition_raw: 0=闭合, 1=打开
    数据侧同样是 0=闭合, 1=打开，因此不做反向，仅做裁剪
    """
    return float(np.clip(clamp_val, 0.0, 1.0))


def main():
    parser = argparse.ArgumentParser(
        description="单臂轨迹复现（Startouch + merged_trajectory.txt + clamp_data_tum.txt）"
    )
    parser.add_argument("--traj", type=str, default="",
                        help="merged_trajectory.txt 路径（不填则交互选择）")
    parser.add_argument("--clamp", type=str, default="",
                        help="clamp_data_tum.txt 路径（不填则交互选择）")
    parser.add_argument("--data_root", type=str, default="~/fastumi/DATA",
                        help="自动搜索数据根目录（默认 ~/fastumi/DATA）")
    parser.add_argument("--clamp_norm_mode", type=str, default="auto",
                        choices=["auto", "minmax", "none"],
                        help="夹爪归一化模式：auto/minmax/none（默认 auto）")
    parser.add_argument("--can_interface", type=str, default="can0",
                        help="Startouch CAN 接口名称")
    parser.add_argument("--enable_fd", action="store_true",
                        help="启用 CAN FD（默认关闭）")
    parser.add_argument("--base_pose", type=float, nargs=6,
                        default=[0.3, 0.0, 0.16, 0.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                        help="基座位姿 [x y z roll pitch yaw]，xyz 单位米，rpy 单位度")

    parser.add_argument("--dt", type=float, default=0.00,
                        help="调度落后时的最小间隔秒数")
    parser.add_argument("--step_interval", type=int, default=2,
                        help="每隔多少帧执行一次（默认 1，即逐帧）")
    parser.add_argument("--speed_rate", type=float, default=1.0,
                        help="回放倍速（>1 更快，<1 更慢）")
    parser.add_argument("--log_interval", type=int, default=20,
                        help="每隔多少步打印一次日志")
    parser.add_argument("--gripper_interval", type=int, default=10,
                        help="每隔多少步发送一次夹爪命令")
    parser.add_argument("--gripper_deadband", type=float, default=0.02,
                        help="夹爪变化阈值，变化小于该值时不重复发送")
    parser.add_argument("--no_gripper", action="store_true",
                        help="不发送夹爪指令")
    parser.add_argument("--dry_run", action="store_true",
                        help="只打印位姿，不连机器人")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="最大复现步数，默认全部")
    args = parser.parse_args()

    if not args.traj:
        args.traj, args.clamp = auto_select_data_paths(args.data_root)
    elif not args.clamp:
        # 仅传 traj 时，按约定自动推导 clamp 路径
        session_dir = os.path.dirname(os.path.dirname(args.traj))
        args.clamp = os.path.join(session_dir, "Clamp_Data", "clamp_data_tum.txt")

    bx, by, bz, br, bp, byw = args.base_pose
    T_base = build_T_base(bx, by, bz, br, bp, byw)
    print(f"base_pose: pos=[{bx:.4f},{by:.4f},{bz:.4f}] m, rpy=[{br:.2f},{bp:.2f},{byw:.2f}] deg")

    print(f"加载轨迹: {args.traj}")
    traj_ts, poses = load_trajectory(args.traj)
    n_total = len(traj_ts)
    print(f"  共 {n_total} 帧，时间范围 [{traj_ts[0]:.3f}, {traj_ts[-1]:.3f}]")

    use_gripper = not args.no_gripper
    clamp_interp = None
    if use_gripper and args.clamp and os.path.isfile(args.clamp):
        print(f"加载夹爪: {args.clamp}")
        clamp_ts, clamp_vals = load_clamp(args.clamp)
        print(f"  共 {len(clamp_ts)} 帧，值域 [{clamp_vals.min():.2f}, {clamp_vals.max():.2f}]")
        clamp_interp = interpolate_clamp(traj_ts, clamp_ts, clamp_vals)
        clamp_interp, norm_changed, norm_info = normalize_clamp_values(
            clamp_interp, mode=args.clamp_norm_mode
        )
        print(f"  归一化: {norm_info}")
        if norm_changed:
            print(f"  归一化后值域 [{clamp_interp.min():.4f}, {clamp_interp.max():.4f}]")
    else:
        if use_gripper:
            print("未找到夹爪文件，跳过夹爪控制。")
        use_gripper = False

    if args.max_steps is not None:
        n_total = min(n_total, args.max_steps)
    frame_indices = list(range(0, n_total, args.step_interval))
    n_steps = len(frame_indices)
    print(f"实际执行步数: {n_steps}（间隔 {args.step_interval}）")

    st_poses = []
    st_grips = []
    for idx in frame_indices:
        st_poses.append(pose7_to_startouch(poses[idx], T_base))
        if use_gripper:
            st_grips.append(clamp_to_startouch(clamp_interp[idx]))
        else:
            st_grips.append(None)

    if args.dry_run:
        print("\n[dry_run] 打印前 10 步：")
        for i in range(min(10, n_steps)):
            fi = frame_indices[i]
            pos, euler = st_poses[i]
            g = st_grips[i]
            print(f"  Step {i:4d} frame={fi:5d} ts={traj_ts[fi]:.3f} "
                  f"pos=[{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}] "
                  f"rpy=[{euler[0]:.5f},{euler[1]:.5f},{euler[2]:.5f}] "
                  f"grip={g}")
        print("... (dry_run 完成)")
        return

    print(f"\n连接 Startouch: can={args.can_interface}, enable_fd={args.enable_fd}")
    arm = SingleArm(can_interface_=args.can_interface, enable_fd_=args.enable_fd, gripper=use_gripper)
    time.sleep(0.2)

    # 与 xarm 版本一致：回放前先到 base_pose 定义的初始位姿（规划模式）
    home_pos = [bx, by, bz]
    home_euler = np.radians([br, bp, byw]).tolist()
    print(f"回初始位姿(base_pose): pos={home_pos}, rpy_rad={[round(v, 6) for v in home_euler]}")
    arm.set_end_effector_pose_euler(pos=home_pos, euler=home_euler, tf=1.0)
    if use_gripper:
        arm.setGripperPosition_raw(1.0)
        print("夹爪初始值置为 1.0（完全张开）")
    time.sleep(3.2)

    print("开始复现轨迹（_raw 透传模式）...")
    replay_ts = (traj_ts[frame_indices] - traj_ts[frame_indices[0]]) / max(args.speed_rate, 1e-6)
    start_time = time.time()
    last_gripper_cmd = None
    try:
        for i in range(n_steps):
            fi = frame_indices[i]
            pos, euler = st_poses[i]
            g = st_grips[i]

            target_t = replay_ts[i]
            now_t = time.time() - start_time
            wait_t = target_t - now_t
            if wait_t > 0:
                time.sleep(wait_t)
            elif args.dt > 0:
                time.sleep(args.dt)

            if i % max(args.log_interval, 1) == 0 or i == n_steps - 1:
                print(f"  [{i+1:4d}/{n_steps}] frame={fi:5d} "
                      f"pos=[{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}] "
                      f"rpy=[{euler[0]:.5f},{euler[1]:.5f},{euler[2]:.5f}] "
                      f"grip={g}")

            arm.set_end_effector_pose_euler_raw(pos=pos, euler=euler)

            if use_gripper and g is not None:
                need_send_gripper = (
                    last_gripper_cmd is None
                    or abs(g - last_gripper_cmd) >= args.gripper_deadband
                    or i % max(args.gripper_interval, 1) == 0
                )
                if need_send_gripper:
                    arm.setGripperPosition_raw(g)
                    last_gripper_cmd = g

        print("轨迹复现完成。")
    finally:
        try:
            print("回放结束，返回初始位姿(base_pose)...")
            arm.set_end_effector_pose_euler(pos=home_pos, euler=home_euler, tf=1.0)
            time.sleep(3.2)
        finally:
            arm.cleanup()


if __name__ == "__main__":
    main()

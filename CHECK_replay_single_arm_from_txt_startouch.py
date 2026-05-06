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
import numpy as np
from scipy.spatial.transform import Rotation

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STARTOUCH_INTERFACE_DIR = os.path.join(SCRIPT_DIR, "startouch-v1", "interface_py")
if STARTOUCH_INTERFACE_DIR not in sys.path:
    sys.path.append(STARTOUCH_INTERFACE_DIR)

from startouchclass import SingleArm, quaternion_to_euler_xyzw


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
    parser.add_argument("--traj", type=str,
                        default="/home/alanliu/fastumi/DATA/task_20260506Z20260506Z20260506_zhantingcaiji/background/multi_session_20260506/session_135658/Merged_Trajectory/merged_trajectory.txt",
                        help="merged_trajectory.txt 路径")
    parser.add_argument("--clamp", type=str,
                        default="/home/alanliu/fastumi/DATA/task_20260506Z20260506Z20260506_zhantingcaiji/background/multi_session_20260506/session_135658/Clamp_Data/clamp_data_tum.txt",
                        help="clamp_data_tum.txt 路径（留空则不控制夹爪）")
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
    parser.add_argument("--speed_rate", type=float, default=2.0,
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

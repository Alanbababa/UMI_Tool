#!/usr/bin/env python3
"""
单臂轨迹复现（TXT 数据源）
=========================

本脚本用于从 TXT 文件读取末端位姿与夹爪数据，按原始时间轴回放到 xArm。
当前实现使用 xArm servo mode + set_servo_cartesian 连续发送笛卡尔目标，
相较逐点 set_position 更平滑，适合高频轨迹复现。

一、输入文件格式
---------------
1) merged_trajectory.txt（必需）
   每行 8 列：
     timestamp  x_m  y_m  z_m  qx  qy  qz  qw

2) clamp_data_tum.txt（可选）
   每行 2 列：
     timestamp  clamp_value

其中：
- timestamp 单位为秒（浮点）
- 位姿位置单位为米，四元数为 [qx, qy, qz, qw]
- clamp_value 为传感器原始夹爪值（常见范围 0~88）

二、坐标与单位转换
-----------------
- 轨迹位姿默认在局部传感器坐标系下，需要转换到机器人基座坐标系后再发送
- 位置：p_base = p_local + base_origin
- 旋转：R_base = R_local @ R_base_frame
- 发送到 xArm 前执行：
  - 位置：m -> mm
  - 欧拉角：rad -> deg（发送时 is_radian=False）

三、快速开始
-----------
最小可用命令：
python TEST_replay_single_arm_from_txt.py \
  --robot_ip 192.168.1.240 \
  --traj /path/to/merged_trajectory.txt \
  --clamp /path/to/clamp_data_tum.txt

推荐平滑参数：
python TEST_replay_single_arm_from_txt.py \
  --robot_ip 192.168.1.240 \
  --traj /path/to/merged_trajectory.txt \
  --clamp /path/to/clamp_data_tum.txt \
  --step_interval 1 \
  --speed_rate 1.0 \
  --log_interval 20 \
  --gripper_interval 10 \
  --gripper_deadband 2

仅检查前几步（不连接机器人）：
python TEST_replay_single_arm_from_txt.py --traj /path/to/merged_trajectory.txt --dry_run

四、参数说明
-----------
--robot_ip str
  xArm IP 地址。

--traj str
  轨迹文件路径（merged_trajectory.txt）。

--clamp str
  夹爪文件路径（clamp_data_tum.txt），不存在时自动跳过夹爪控制。

--base_pose float x6
  基座/回 Home 位姿：[x y z roll pitch yaw]。
  xyz 单位米，rpy 单位度。该参数同时用于坐标转换基准。

--gripper_open_val float
  传感器值对应“夹爪完全打开”（默认 88）。

--gripper_closed_val float
  传感器值对应“夹爪完全闭合”（默认 0）。

--dt float
  最小循环间隔（秒）。当调度落后时用于防止指令突发堆积（默认 0.01）。

--step_interval int
  轨迹抽帧步长。1 为逐帧，2 为隔帧，值越大越不平滑但负载更低（默认 1）。

--speed_rate float
  回放倍速。1.0 为原速，>1 更快，<1 更慢（默认 1.0）。

--log_interval int
  每隔多少步打印一条状态日志（默认 20）。

--gripper_interval int
  每隔多少步至少发送一次夹爪指令（默认 10）。

--gripper_deadband int
  夹爪命令死区：新旧命令差值小于该阈值时不重复发送（默认 2）。

--no_gripper
  禁用夹爪发送。

--dry_run
  仅打印转换后的轨迹，不连接机器人。

--max_steps int
  最大回放步数，便于调试。

五、平滑性调参建议
-----------------
- 优先保证 step_interval=1；如果设备负载高，再尝试 2
- 出现跟踪抖动时，先把 speed_rate 降到 0.8~0.9
- 夹爪动作频繁会影响主循环，适当增大 gripper_interval 或 deadband
- 日志过多会拖慢循环，保持 log_interval >= 20

六、常见问题
-----------
1) 机械臂“顿挫/一卡一卡”
   - 通常由抽帧过大（step_interval 过高）或时间调度落后造成
   - 建议 step_interval=1、speed_rate<=1.0，并减少日志与夹爪发送频率

2) 姿态/位置方向不对
   - 优先检查 base_pose 是否与采集时的基座定义一致

3) 夹爪开合方向反了
   - 调整 gripper_open_val / gripper_closed_val 的对应关系
"""
import argparse
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def load_trajectory(txt_path):
    """
    加载 merged_trajectory.txt。
    返回: timestamps (N,), poses (N, 7) = [x_m, y_m, z_m, qx, qy, qz, qw]
    """
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f'轨迹文件不存在: {txt_path}')
    data = np.loadtxt(txt_path)          # (N, 8)
    timestamps = data[:, 0]
    poses = data[:, 1:]                  # (N, 7): x y z qx qy qz qw
    return timestamps, poses


def load_clamp(clamp_path):
    """
    加载 clamp_data_tum.txt。
    返回: timestamps (M,), values (M,)
    """
    if not os.path.isfile(clamp_path):
        return None, None
    data = np.loadtxt(clamp_path)        # (M, 2)
    return data[:, 0], data[:, 1]


def interpolate_clamp(traj_ts, clamp_ts, clamp_vals):
    """
    将夹爪时间序列插值到轨迹时间戳上。
    clamp_ts/vals 必须按时间升序排列。
    """
    return np.interp(traj_ts, clamp_ts, clamp_vals)


# ── 坐标系转换 ────────────────────────────────────────────────────────────────

def build_T_base(base_x, base_y, base_z, base_roll_deg, base_pitch_deg, base_yaw_deg):
    """
    构建基座变换矩阵 T_base_to_local（4×4），与 CONVERSION 脚本逻辑一致。
      位置部分: [base_x, base_y, base_z]（米）
      旋转部分: 'xyz' 内旋欧拉角（度）
    """
    rot = Rotation.from_euler('xyz',
                               [base_roll_deg, base_pitch_deg, base_yaw_deg],
                               degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [base_x, base_y, base_z]
    return T


def transform_to_base_quat(x, y, z, qx, qy, qz, qw, T_base):
    """
    将局部传感器坐标系位姿变换到机器人基座坐标系。
    与 CONVERSION_data_process_tcp_step1.py::transform_to_base_quat 完全一致：
      位置: p_base = base_origin + p_local
      旋转: R_base = R_local @ R_base_frame
    返回: x_base, y_base, z_base, roll_base_rad, pitch_base_rad, yaw_base_rad
    """
    R_local = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    R_base_frame = T_base[:3, :3]
    base_origin = T_base[:3, 3]

    R_base = R_local @ R_base_frame
    x_base, y_base, z_base = base_origin + np.array([x, y, z])

    roll, pitch, yaw = Rotation.from_matrix(R_base).as_euler('xyz', degrees=False)
    return x_base, y_base, z_base, roll, pitch, yaw


# ── 转換到 xarm 指令 ──────────────────────────────────────────────────────────

def pose7_to_xarm(pose7, T_base):
    """
    pose7: [x_m, y_m, z_m, qx, qy, qz, qw]（传感器局部坐标系）
    1. 坐标系变换到机器人基座系
    2. 米 → 毫米，弧度 → 度
    返回: [x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
    """
    x, y, z, qx, qy, qz, qw = pose7
    x_b, y_b, z_b, roll_r, pitch_r, yaw_r = transform_to_base_quat(
        x, y, z, qx, qy, qz, qw, T_base
    )
    return [x_b * 1000.0, y_b * 1000.0, z_b * 1000.0,
            np.degrees(roll_r), np.degrees(pitch_r), np.degrees(yaw_r)]


def clamp_to_robotiq(clamp_val, open_val, closed_val):
    """
    将传感器夹爪值映射到 robotiq 0~255（0=开，255=闭）。
    open_val  : 传感器读数对应"完全打开"
    closed_val: 传感器读数对应"完全闭合"
    """
    ratio = (clamp_val - open_val) / (closed_val - open_val + 1e-9)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    return int(ratio * 255)


# ── Home 位姿 ─────────────────────────────────────────────────────────────────



# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='单臂轨迹复现（merged_trajectory.txt + clamp_data_tum.txt）'
    )
    parser.add_argument('--robot_ip', type=str, default='192.168.1.238',
                        help='机器人 IP 地址')
    parser.add_argument('--traj', type=str,
                        default='/home/ubuntu/replay/Data/session_2/merged_trajectory.txt',
                        help='merged_trajectory.txt 路径')
    parser.add_argument('--clamp', type=str,
                        default='/home/ubuntu/replay/Data/session_2/clamp_data_tum.txt',
                        help='clamp_data_tum.txt 路径（留空则不控制夹爪）')

    parser.add_argument('--base_pose', type=float, nargs=6,
                        default=[0.47, 0.0, 0.145, 180.0, -90.0, 0.0],
                        metavar=('X', 'Y', 'Z', 'ROLL', 'PITCH', 'YAW'),
                        help='Home/基座位姿 [x y z roll pitch yaw]，xyz 单位米，rpy 单位度')

    parser.add_argument('--gripper_open_val', type=float, default=86.0,
                        help='传感器值对应夹爪完全打开（默认 88）')
    parser.add_argument('--gripper_closed_val', type=float, default=0.0,
                        help='传感器值对应夹爪完全闭合（默认 0）')
    parser.add_argument('--dt', type=float, default=0.001,
                        help='每步间隔秒数（默认 0.01）')
    parser.add_argument('--step_interval', type=int, default=1,
                        help='每隔多少帧执行一次（默认 1，即逐帧）')
    parser.add_argument('--speed_rate', type=float, default=2.0,
                        help='回放倍速（>1 更快，<1 更慢，默认 1.0）')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='每隔多少步打印一次日志（默认 20）')
    parser.add_argument('--gripper_interval', type=int, default=10,
                        help='每隔多少步发送一次夹爪命令（默认 10）')
    parser.add_argument('--gripper_deadband', type=int, default=2,
                        help='夹爪变化阈值，变化小于该值时不重复发送（默认 2）')
    parser.add_argument('--no_gripper', action='store_true',
                        help='不发送夹爪指令')
    parser.add_argument('--dry_run', action='store_true',
                        help='只打印位姿，不连机器人')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='最大复现步数，默认全部')
    args = parser.parse_args()

    # base_pose 同时作为 Home 位姿和坐标系变换基准
    bx, by, bz, br, bp, byw = args.base_pose
    ARM_HOME_POSE = [bx * 1000, by * 1000, bz * 1000, br, bp, byw]
    T_base = build_T_base(bx, by, bz, br, bp, byw)
    print(f'Home/基座: [{bx*1000:.1f}, {by*1000:.1f}, {bz*1000:.1f}] mm, euler=[{br}, {bp}, {byw}] deg')

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    print(f'加载轨迹: {args.traj}')
    traj_ts, poses = load_trajectory(args.traj)
    n_total = len(traj_ts)
    print(f'  共 {n_total} 帧，时间范围 [{traj_ts[0]:.3f}, {traj_ts[-1]:.3f}]')

    use_gripper = not args.no_gripper
    clamp_interp = None
    if use_gripper and args.clamp and os.path.isfile(args.clamp):
        print(f'加载夹爪: {args.clamp}')
        clamp_ts, clamp_vals = load_clamp(args.clamp)
        print(f'  共 {len(clamp_ts)} 帧，值域 [{clamp_vals.min():.2f}, {clamp_vals.max():.2f}]')
        clamp_interp = interpolate_clamp(traj_ts, clamp_ts, clamp_vals)
    else:
        if use_gripper:
            print('未找到夹爪文件，跳过夹爪控制。')
        use_gripper = False

    # ── 确定执行帧 ────────────────────────────────────────────────────────────
    if args.max_steps is not None:
        n_total = min(n_total, args.max_steps)
    frame_indices = list(range(0, n_total, args.step_interval))
    n_steps = len(frame_indices)
    print(f'实际执行步数: {n_steps}（间隔 {args.step_interval}）')

    # ── 预转换所有位姿 ────────────────────────────────────────────────────────
    xarm_poses = []
    robotiq_grips = []
    for idx in frame_indices:
        xarm_poses.append(pose7_to_xarm(poses[idx], T_base))
        if use_gripper:
            robotiq_grips.append(
                clamp_to_robotiq(clamp_interp[idx],
                                 args.gripper_open_val,
                                 args.gripper_closed_val)
            )
        else:
            robotiq_grips.append(None)

    # ── dry_run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print('\n[dry_run] 打印前 10 步：')
        for i in range(min(10, n_steps)):
            fi = frame_indices[i]
            p = xarm_poses[i]
            g = robotiq_grips[i]
            print(f'  Step {i:4d} frame={fi:5d} ts={traj_ts[fi]:.3f} '
                  f'pos=[{p[0]:7.2f},{p[1]:7.2f},{p[2]:7.2f}] '
                  f'rpy=[{p[3]:7.2f},{p[4]:7.2f},{p[5]:7.2f}] '
                  f'grip={g}')
        print('... (dry_run 完成)')
        return

    # ── 连接机器人 ────────────────────────────────────────────────────────────
    print(f'\n连接机器人: {args.robot_ip}')
    arm = XArmAPI(args.robot_ip)
    time.sleep(0.5)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(0)

    # 回 Home
    print(f'回 Home: {ARM_HOME_POSE}')
    arm.set_position(*ARM_HOME_POSE, wait=True)
    if use_gripper:
        arm.robotiq_set_position(0, wait=False)
    time.sleep(1.0)

    # 切换到伺服模式做连续笛卡尔流控，减少逐点 set_position 的停顿感
    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(0.1)

    # ── 执行轨迹 ──────────────────────────────────────────────────────────────
    print('开始复现轨迹...')
    replay_ts = (traj_ts[frame_indices] - traj_ts[frame_indices[0]]) / max(args.speed_rate, 1e-6)
    start_time = time.time()
    last_gripper_cmd = None
    for i in range(n_steps):
        fi = frame_indices[i]
        p = xarm_poses[i]
        g = robotiq_grips[i]
        target_t = replay_ts[i]
        now_t = time.time() - start_time
        wait_t = target_t - now_t
        if wait_t > 0:
            time.sleep(wait_t)
        elif args.dt > 0:
            # 落后时仍保留最小间隔，避免指令突发堆积导致抖动
            time.sleep(args.dt)

        if i % max(args.log_interval, 1) == 0 or i == n_steps - 1:
            print(f'  [{i+1:4d}/{n_steps}] frame={fi:5d} '
                  f'pos=[{p[0]:7.2f},{p[1]:7.2f},{p[2]:7.2f}] '
                  f'rpy=[{p[3]:7.2f},{p[4]:7.2f},{p[5]:7.2f}] '
                  f'grip={g}')

        arm.set_servo_cartesian(p, is_radian=False)

        if use_gripper and g is not None:
            need_send_gripper = (
                last_gripper_cmd is None
                or abs(g - last_gripper_cmd) >= args.gripper_deadband
                or i % max(args.gripper_interval, 1) == 0
            )
            if need_send_gripper:
                arm.robotiq_set_position(g, wait=False)
                last_gripper_cmd = g

    print('轨迹复现完成。')

    arm.set_mode(0)
    arm.set_state(0)
    # 回 Home
    print(f'回 Home: {ARM_HOME_POSE}')
    arm.set_position(*ARM_HOME_POSE, wait=True)
    if use_gripper:
        arm.robotiq_set_position(0, wait=False)
    time.sleep(1.0)


if __name__ == '__main__':
    main()

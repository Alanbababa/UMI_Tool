#!/usr/bin/env python3
"""
Startouch 单臂实时遥操作（ROS topic 数据源）
=========================================

订阅本机 ROS1 topic：
- /xv_sdk/250801DR48FP25002352/slam/pose
  支持示例中的自定义消息结构：confidence + poseMsg.pose
- /xv_sdk/250801DR48FP25002352/clamp/Data
  支持示例中的 data 字段，默认把 0~88 映射到 Startouch 夹爪 raw 值 0~1

控制端沿用 replay 脚本里的 Startouch _raw 透传接口：
- SingleArm.set_end_effector_pose_euler_raw(pos, euler)
- SingleArm.setGripperPosition_raw(value)

默认流程：
1. 读取 SLAM 位姿 [x, y, z, qx, qy, qz, qw]
2. 按 XV2Gripper 进行坐标系和夹爪安装偏移转换
3. 第一帧自动定零，机械臂从 --base_pose 开始跟随相对运动
4. 高频循环发送末端位姿，低频/死区发送夹爪
"""

import argparse
import os
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STARTOUCH_CANDIDATE_DIRS = [
    os.environ.get("STARTOUCH_INTERFACE_DIR", ""),
    os.path.join(SCRIPT_DIR, "startouch_sdk", "interface_py"),
    os.path.join(SCRIPT_DIR, "startouch-v1", "interface_py"),
    os.path.expanduser("~/startouch_sdk/interface_py"),
    os.path.expanduser("~/startouch-v1/interface_py"),
]
for _path in STARTOUCH_CANDIDATE_DIRS:
    if _path and os.path.isdir(_path) and _path not in sys.path:
        sys.path.append(_path)

try:
    from startouchclass import SingleArm, quaternion_to_euler_xyzw
    STARTOUCH_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - 依赖实际机器人 SDK 环境
    SingleArm = None
    STARTOUCH_IMPORT_ERROR = exc

    def quaternion_to_euler_xyzw(q_xyzw):
        return Rotation.from_quat(q_xyzw).as_euler("xyz", degrees=False)


DEFAULT_POSE_TOPIC = "/xv_sdk/250801DR48FP25002352/slam/pose"
DEFAULT_CLAMP_TOPIC = "/xv_sdk/250801DR48FP25002352/clamp/Data"


def normalize_quat(q_xyzw):
    q = np.asarray(q_xyzw, dtype=float)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        raise ValueError("四元数模长过小，无法归一化")
    return q / norm


def qpos2mat(qpos):
    """[x, y, z, qx, qy, qz, qw] -> 4x4 齐次矩阵。"""
    qpos = np.asarray(qpos, dtype=float)
    T = np.eye(4)
    T[:3, 3] = qpos[:3]
    T[:3, :3] = Rotation.from_quat(normalize_quat(qpos[3:7])).as_matrix()
    return T


def mat2qpos(T):
    """4x4 齐次矩阵 -> [x, y, z, qx, qy, qz, qw]。"""
    qpos = np.zeros(7, dtype=float)
    qpos[:3] = T[:3, 3]
    qpos[3:7] = Rotation.from_matrix(T[:3, :3]).as_quat()
    return qpos


def XV2Gripper(qpos, left_right_offset, front_back_offset, up_down_offset):
    """
    将 XV SLAM 位姿转换到夹爪参考系。

    逻辑按用户提供的 XV2Gripper 保持一致；输入/输出均为
    [x, y, z, qx, qy, qz, qw]，位置单位米，四元数顺序 xyzw。
    """
    transformation_mat = np.array([
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=float)

    cur_qpos = np.asarray(qpos, dtype=float).copy()
    cur_mat = qpos2mat(cur_qpos)

    cur_qpos[0] -= left_right_offset
    cur_qpos[1] -= up_down_offset
    cur_qpos[2] -= front_back_offset

    ori = cur_mat[:3, :3]
    cur_qpos[:3] += ori[:, 0] * left_right_offset
    cur_qpos[:3] += ori[:, 1] * up_down_offset
    cur_qpos[:3] += ori[:, 2] * front_back_offset

    cur_mat = qpos2mat(cur_qpos)
    xv_mat_in_gripper = transformation_mat @ cur_mat @ np.linalg.inv(transformation_mat)
    return mat2qpos(xv_mat_in_gripper)


def build_T_base(base_x, base_y, base_z, base_roll_deg, base_pitch_deg, base_yaw_deg):
    """构建 Startouch base/home 位姿矩阵，xyz 单位米，rpy 单位度。"""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler(
        "xyz",
        [base_roll_deg, base_pitch_deg, base_yaw_deg],
        degrees=True,
    ).as_matrix()
    T[:3, 3] = [base_x, base_y, base_z]
    return T


def source_qpos_to_robot_matrix(qpos, T_base):
    """
    与现有 Startouch replay 脚本保持一致：
    p_robot = base_origin + p_source
    R_robot = R_source @ R_base_frame
    """
    qpos = np.asarray(qpos, dtype=float)
    T = np.eye(4)
    T[:3, 3] = T_base[:3, 3] + qpos[:3]
    R_source = Rotation.from_quat(normalize_quat(qpos[3:7])).as_matrix()
    T[:3, :3] = R_source @ T_base[:3, :3]
    return T


def scale_delta_matrix(T_delta, position_scale, rotation_scale):
    """对相对位移和相对姿态做倍率缩放。"""
    T_scaled = np.eye(4)
    T_scaled[:3, 3] = T_delta[:3, 3] * position_scale
    rotvec = Rotation.from_matrix(T_delta[:3, :3]).as_rotvec()
    T_scaled[:3, :3] = Rotation.from_rotvec(rotvec * rotation_scale).as_matrix()
    return T_scaled


def blend_matrices(previous_T, target_T, alpha):
    """位姿一阶低通；alpha=1 表示不滤波。"""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if previous_T is None or alpha >= 1.0:
        return target_T
    if alpha <= 0.0:
        return previous_T

    out = np.eye(4)
    out[:3, 3] = (1.0 - alpha) * previous_T[:3, 3] + alpha * target_T[:3, 3]
    rots = Rotation.from_matrix(np.stack([previous_T[:3, :3], target_T[:3, :3]]))
    out[:3, :3] = Slerp([0, 1], rots)([alpha])[0].as_matrix()
    return out


def matrix_to_startouch(T):
    """4x4 robot 矩阵 -> Startouch raw 接口所需 pos + euler(rad)。"""
    pos = T[:3, 3].astype(float).tolist()
    q_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    euler = np.asarray(quaternion_to_euler_xyzw(q_xyzw), dtype=float).tolist()
    return pos, euler


def clamp_to_startouch(raw_value, open_val, closed_val, invert=False):
    """将 ROS 夹爪读数映射到 Startouch raw [0, 1]，默认 0=闭合，1=打开。"""
    denom = float(open_val - closed_val)
    if abs(denom) < 1e-9:
        raise ValueError("gripper_open_val 和 gripper_closed_val 不能相同")
    value = (float(raw_value) - float(closed_val)) / denom
    value = float(np.clip(value, 0.0, 1.0))
    return 1.0 - value if invert else value


def stamp_to_sec(stamp):
    if stamp is None:
        return None
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    secs = getattr(stamp, "secs", None)
    nsecs = getattr(stamp, "nsecs", 0)
    if secs is None:
        return None
    return float(secs) + float(nsecs) * 1e-9


def get_msg_stamp(msg):
    candidates = [msg]
    pose_msg = getattr(msg, "poseMsg", None)
    if pose_msg is not None:
        candidates.insert(0, pose_msg)
    for obj in candidates:
        header = getattr(obj, "header", None)
        if header is not None:
            stamp = stamp_to_sec(getattr(header, "stamp", None))
            if stamp is not None:
                return stamp
    timestamp = getattr(msg, "timestamp", None)
    if timestamp is not None:
        try:
            return float(timestamp)
        except (TypeError, ValueError):
            return None
    return None


def extract_pose_obj(msg):
    """
    兼容以下结构：
    - custom.poseMsg.pose
    - PoseStamped.pose
    - Odometry.pose.pose
    - Pose
    """
    roots = []
    pose_msg = getattr(msg, "poseMsg", None)
    if pose_msg is not None:
        roots.append(pose_msg)
    roots.append(msg)

    for root in roots:
        pose = getattr(root, "pose", None)
        if pose is not None:
            if hasattr(pose, "pose"):
                pose = pose.pose
            if hasattr(pose, "position") and hasattr(pose, "orientation"):
                return pose
        if hasattr(root, "position") and hasattr(root, "orientation"):
            return root

    raise ValueError("无法从消息中找到 pose.position / pose.orientation")


def extract_pose7(msg):
    pose = extract_pose_obj(msg)
    p = pose.position
    q = pose.orientation
    return np.array([
        float(p.x),
        float(p.y),
        float(p.z),
        float(q.x),
        float(q.y),
        float(q.z),
        float(q.w),
    ], dtype=float)


def extract_gripper_value(msg):
    value = getattr(msg, "data", None)
    if value is None:
        value = getattr(msg, "value", None)
    if hasattr(value, "data"):
        value = value.data
    if value is None:
        raise ValueError("无法从夹爪消息中找到 data/value 字段")
    return float(value)


def extract_confidence(msg):
    confidence = getattr(msg, "confidence", None)
    if confidence is None:
        return None
    try:
        return float(confidence)
    except (TypeError, ValueError):
        return None


class LatestRosState:
    def __init__(self, min_confidence):
        self.min_confidence = min_confidence
        self.lock = threading.Lock()
        self.pose_qpos = None
        self.pose_stamp = None
        self.pose_recv_time = None
        self.pose_seq = 0
        self.pose_confidence = None
        self.pose_error_count = 0
        self.gripper_raw = None
        self.gripper_stamp = None
        self.gripper_recv_time = None
        self.gripper_seq = 0
        self.gripper_error_count = 0

    def update_pose(self, msg):
        try:
            confidence = extract_confidence(msg)
            if confidence is not None and confidence < self.min_confidence:
                return
            qpos = extract_pose7(msg)
            stamp = get_msg_stamp(msg)
        except Exception as exc:
            self.pose_error_count += 1
            if self.pose_error_count <= 3:
                print(f"[pose] 解析失败({self.pose_error_count}): {exc}")
            return

        with self.lock:
            self.pose_qpos = qpos
            self.pose_stamp = stamp
            self.pose_recv_time = time.time()
            self.pose_seq += 1
            self.pose_confidence = confidence

    def update_gripper(self, msg):
        try:
            value = extract_gripper_value(msg)
            stamp = get_msg_stamp(msg)
        except Exception as exc:
            self.gripper_error_count += 1
            if self.gripper_error_count <= 3:
                print(f"[gripper] 解析失败({self.gripper_error_count}): {exc}")
            return

        with self.lock:
            self.gripper_raw = value
            self.gripper_stamp = stamp
            self.gripper_recv_time = time.time()
            self.gripper_seq += 1

    def snapshot(self):
        with self.lock:
            return {
                "pose_qpos": None if self.pose_qpos is None else self.pose_qpos.copy(),
                "pose_stamp": self.pose_stamp,
                "pose_recv_time": self.pose_recv_time,
                "pose_seq": self.pose_seq,
                "pose_confidence": self.pose_confidence,
                "gripper_raw": self.gripper_raw,
                "gripper_stamp": self.gripper_stamp,
                "gripper_recv_time": self.gripper_recv_time,
                "gripper_seq": self.gripper_seq,
            }


def resolve_topic_class(rospy, rostopic, topic, timeout):
    """等待 publisher 出现并解析 topic 消息类型。"""
    start = time.time()
    while not rospy.is_shutdown():
        msg_cls, real_topic, _ = rostopic.get_topic_class(topic, blocking=False)
        if msg_cls is not None:
            return msg_cls, real_topic or topic
        if timeout is not None and timeout > 0 and time.time() - start > timeout:
            raise TimeoutError(f"等待 topic 超时: {topic}")
        rospy.sleep(0.1)
    raise RuntimeError("ROS 已关闭，无法解析 topic")


def maybe_clip_workspace(T, workspace_min, workspace_max, enable_clip):
    if not enable_clip:
        return T, False
    clipped = np.clip(T[:3, 3], workspace_min, workspace_max)
    changed = bool(np.max(np.abs(clipped - T[:3, 3])) > 1e-9)
    if changed:
        T = T.copy()
        T[:3, 3] = clipped
    return T, changed


def connect_startouch(args, use_gripper):
    if SingleArm is None:
        raise RuntimeError(
            "无法导入 Startouch SDK 的 startouchclass。请检查 startouch_sdk/interface_py "
            f"或设置 STARTOUCH_INTERFACE_DIR。原始错误: {STARTOUCH_IMPORT_ERROR}"
        )

    print(f"连接 Startouch: can={args.can_interface}, enable_fd={args.enable_fd}")
    arm = SingleArm(
        can_interface_=args.can_interface,
        enable_fd_=args.enable_fd,
        gripper=use_gripper,
    )
    time.sleep(0.2)
    return arm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Startouch 单臂实时遥操作：ROS SLAM pose + clamp -> _raw 控制"
    )
    parser.add_argument("--pose_topic", type=str, default=DEFAULT_POSE_TOPIC,
                        help="SLAM 位姿 topic")
    parser.add_argument("--clamp_topic", type=str, default=DEFAULT_CLAMP_TOPIC,
                        help="夹爪 topic")
    parser.add_argument("--node_name", type=str, default="startouch_ros_teleop",
                        help="ROS 节点名")
    parser.add_argument("--topic_wait_timeout", type=float, default=10.0,
                        help="等待 topic publisher 的超时时间；<=0 表示一直等待")
    parser.add_argument("--rate", type=float, default=30.0,
                        help="机械臂位姿发送频率 Hz")
    parser.add_argument("--log_period", type=float, default=1.0,
                        help="状态日志周期，秒")
    parser.add_argument("--min_confidence", type=float, default=0.5,
                        help="pose 消息 confidence 低于该值时丢弃；无 confidence 字段则不检查")
    parser.add_argument("--max_pose_age", type=float, default=0.5,
                        help="最新 pose 超过该秒数未更新时暂停发送")
    parser.add_argument("--max_gripper_age", type=float, default=2.0,
                        help="最新夹爪数据超过该秒数后不再继续发送")

    parser.add_argument("--can_interface", type=str, default="can0",
                        help="Startouch CAN 接口名称")
    parser.add_argument("--enable_fd", action="store_true",
                        help="启用 CAN FD")
    parser.add_argument("--base_pose", type=float, nargs=6,
                        default=[0.3, 0.0, 0.16, 0.0, 0.0, 0.0],
                        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                        help="机械臂 home/base 位姿 [x y z roll pitch yaw]，xyz 米，rpy 度")
    parser.add_argument("--no_home", action="store_true",
                        help="启动后不先回到 base_pose")
    parser.add_argument("--home_tf", type=float, default=1.0,
                        help="回 base_pose 的规划时间")
    parser.add_argument("--home_wait", type=float, default=3.2,
                        help="回 base_pose 后等待时间")
    parser.add_argument("--return_home", action="store_true",
                        help="退出时回到 base_pose")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--relative", dest="relative", action="store_true", default=True,
                            help="第一帧定零，从 base_pose 开始跟随相对运动（默认）")
    mode_group.add_argument("--absolute", dest="relative", action="store_false",
                            help="不定零，按 replay 逻辑直接 base_pose + 当前 SLAM 位姿")

    parser.add_argument("--no_xv2gripper", action="store_true",
                        help="不执行 XV2Gripper 坐标变换")
    parser.add_argument("--left_right_offset", type=float, default=0.02268,
                        help="XV 到夹爪安装偏移，左右方向，米")
    parser.add_argument("--front_back_offset", type=float, default=0.08745,
                        help="XV 到夹爪安装偏移，前后方向，米")
    parser.add_argument("--up_down_offset", type=float, default=0.09240,
                        help="XV 到夹爪安装偏移，上下方向，米")
    parser.add_argument("--position_scale", type=float, default=1.0,
                        help="相对模式下平移缩放倍率")
    parser.add_argument("--rotation_scale", type=float, default=1.0,
                        help="相对模式下旋转缩放倍率")
    parser.add_argument("--lowpass_alpha", type=float, default=1.0,
                        help="位姿低通滤波 alpha，1=关闭滤波，推荐 0.3~0.8")

    parser.add_argument("--workspace_min", type=float, nargs=3,
                        default=[0.05, -0.35, 0.02],
                        metavar=("X_MIN", "Y_MIN", "Z_MIN"),
                        help="位置安全限幅下界，米")
    parser.add_argument("--workspace_max", type=float, nargs=3,
                        default=[0.65, 0.35, 0.60],
                        metavar=("X_MAX", "Y_MAX", "Z_MAX"),
                        help="位置安全限幅上界，米")
    parser.add_argument("--disable_workspace_clip", action="store_true",
                        help="关闭 workspace 位置限幅")

    parser.add_argument("--no_gripper", action="store_true",
                        help="不订阅/发送夹爪")
    parser.add_argument("--gripper_open_val", type=float, default=88.0,
                        help="ROS 夹爪读数对应完全打开，默认 88")
    parser.add_argument("--gripper_closed_val", type=float, default=0.0,
                        help="ROS 夹爪读数对应完全闭合，默认 0")
    parser.add_argument("--invert_gripper", action="store_true",
                        help="夹爪 raw 输出取反")
    parser.add_argument("--gripper_period", type=float, default=0.1,
                        help="夹爪至少每隔多少秒发送一次")
    parser.add_argument("--gripper_deadband", type=float, default=0.02,
                        help="夹爪 raw 值变化超过该阈值才立即发送")

    parser.add_argument("--dry_run", action="store_true",
                        help="只订阅并打印转换结果，不连接机器人")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import rospy
        import rostopic
    except ImportError as exc:
        raise RuntimeError(
            "需要在 ROS1 Python 环境中运行。请先 source ROS 环境，例如："
            "source /opt/ros/noetic/setup.bash"
        ) from exc

    if args.rate <= 0:
        raise ValueError("--rate 必须大于 0")
    if not (0.0 <= args.lowpass_alpha <= 1.0):
        raise ValueError("--lowpass_alpha 必须在 [0, 1] 内")

    use_gripper = not args.no_gripper
    T_base = build_T_base(*args.base_pose)
    home_pos = args.base_pose[:3]
    home_euler = np.radians(args.base_pose[3:6]).tolist()
    workspace_min = np.asarray(args.workspace_min, dtype=float)
    workspace_max = np.asarray(args.workspace_max, dtype=float)
    enable_workspace_clip = not args.disable_workspace_clip

    rospy.init_node(args.node_name, anonymous=True)
    state = LatestRosState(min_confidence=args.min_confidence)
    topic_timeout = None if args.topic_wait_timeout <= 0 else args.topic_wait_timeout

    print(f"等待 pose topic: {args.pose_topic}")
    pose_cls, pose_real_topic = resolve_topic_class(
        rospy, rostopic, args.pose_topic, topic_timeout
    )
    rospy.Subscriber(pose_real_topic, pose_cls, state.update_pose, queue_size=1)
    print(f"已订阅 pose: {pose_real_topic} ({pose_cls.__name__})")

    if use_gripper:
        print(f"等待 clamp topic: {args.clamp_topic}")
        clamp_cls, clamp_real_topic = resolve_topic_class(
            rospy, rostopic, args.clamp_topic, topic_timeout
        )
        rospy.Subscriber(clamp_real_topic, clamp_cls, state.update_gripper, queue_size=1)
        print(f"已订阅 clamp: {clamp_real_topic} ({clamp_cls.__name__})")

    print(
        "模式: "
        f"{'relative 第一帧定零' if args.relative else 'absolute 绝对位姿'}; "
        f"XV2Gripper={'on' if not args.no_xv2gripper else 'off'}; "
        f"dry_run={args.dry_run}"
    )

    arm = None
    if not args.dry_run:
        arm = connect_startouch(args, use_gripper=use_gripper)
        if not args.no_home:
            print(
                "回 base_pose: "
                f"pos={[round(v, 4) for v in home_pos]}, "
                f"rpy_rad={[round(v, 6) for v in home_euler]}"
            )
            arm.set_end_effector_pose_euler(pos=home_pos, euler=home_euler, tf=args.home_tf)
            time.sleep(args.home_wait)

    rate = rospy.Rate(args.rate)
    source_zero_T = None
    last_cmd_T = None
    last_log_time = 0.0
    last_clip_log_time = 0.0
    last_pose_wait_log_time = 0.0
    last_gripper_cmd = None
    last_gripper_send_time = 0.0

    print("开始实时遥操作。Ctrl-C 退出。")
    try:
        while not rospy.is_shutdown():
            now = time.time()
            snap = state.snapshot()
            pose_qpos = snap["pose_qpos"]

            if pose_qpos is None:
                if now - last_pose_wait_log_time >= 1.0:
                    print("等待第一帧 pose...")
                    last_pose_wait_log_time = now
                rate.sleep()
                continue

            pose_age = now - snap["pose_recv_time"]
            if pose_age > args.max_pose_age:
                if now - last_pose_wait_log_time >= 1.0:
                    print(f"pose 数据过旧({pose_age:.3f}s)，暂停发送。")
                    last_pose_wait_log_time = now
                rate.sleep()
                continue

            if not args.no_xv2gripper:
                source_qpos = XV2Gripper(
                    pose_qpos,
                    left_right_offset=args.left_right_offset,
                    front_back_offset=args.front_back_offset,
                    up_down_offset=args.up_down_offset,
                )
            else:
                source_qpos = pose_qpos

            if args.relative:
                if source_zero_T is None:
                    source_zero_T = qpos2mat(source_qpos)
                    print(
                        "已用第一帧 pose 定零: "
                        f"stamp={snap['pose_stamp']}, "
                        f"qpos={[round(v, 6) for v in source_qpos.tolist()]}"
                    )
                T_delta = np.linalg.inv(source_zero_T) @ qpos2mat(source_qpos)
                T_delta = scale_delta_matrix(
                    T_delta,
                    position_scale=args.position_scale,
                    rotation_scale=args.rotation_scale,
                )
                robot_T = source_qpos_to_robot_matrix(mat2qpos(T_delta), T_base)
            else:
                robot_T = source_qpos_to_robot_matrix(source_qpos, T_base)

            robot_T, clipped = maybe_clip_workspace(
                robot_T,
                workspace_min=workspace_min,
                workspace_max=workspace_max,
                enable_clip=enable_workspace_clip,
            )
            if clipped and now - last_clip_log_time >= 1.0:
                p = robot_T[:3, 3]
                print(
                    "workspace 限幅触发: "
                    f"cmd_pos=[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]"
                )
                last_clip_log_time = now

            robot_T = blend_matrices(last_cmd_T, robot_T, args.lowpass_alpha)
            pos, euler = matrix_to_startouch(robot_T)

            if not args.dry_run:
                arm.set_end_effector_pose_euler_raw(pos=pos, euler=euler)
            last_cmd_T = robot_T

            gripper_cmd = None
            if use_gripper and snap["gripper_raw"] is not None:
                gripper_age = now - snap["gripper_recv_time"]
                if gripper_age <= args.max_gripper_age:
                    gripper_cmd = clamp_to_startouch(
                        snap["gripper_raw"],
                        open_val=args.gripper_open_val,
                        closed_val=args.gripper_closed_val,
                        invert=args.invert_gripper,
                    )
                    should_send = (
                        last_gripper_cmd is None
                        or abs(gripper_cmd - last_gripper_cmd) >= args.gripper_deadband
                        or now - last_gripper_send_time >= args.gripper_period
                    )
                    if should_send:
                        if not args.dry_run:
                            arm.setGripperPosition_raw(gripper_cmd)
                        last_gripper_cmd = gripper_cmd
                        last_gripper_send_time = now

            if now - last_log_time >= args.log_period:
                print(
                    f"[seq={snap['pose_seq']}] "
                    f"pos=[{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}] "
                    f"rpy=[{euler[0]:.5f},{euler[1]:.5f},{euler[2]:.5f}] "
                    f"grip={None if gripper_cmd is None else round(gripper_cmd, 4)} "
                    f"conf={snap['pose_confidence']}"
                )
                last_log_time = now

            rate.sleep()
    finally:
        if arm is not None:
            try:
                if args.return_home:
                    print("退出，回 base_pose...")
                    arm.set_end_effector_pose_euler(
                        pos=home_pos,
                        euler=home_euler,
                        tf=args.home_tf,
                    )
                    time.sleep(args.home_wait)
            finally:
                arm.cleanup()


if __name__ == "__main__":
    main()

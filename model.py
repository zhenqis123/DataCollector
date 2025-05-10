import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import collections
import datetime
import glob
import json
import logging
import os
import pickle
import queue
import selectors
import socket
import struct
import threading
import time
import zipfile
from collections import deque
from queue import Queue

import cv2
import numpy as np
import zstandard
from PyQt5.QtCore import QByteArray, QMutex, QMutexLocker, QThread, pyqtSignal
from tqdm import tqdm  # 需要先安装tqdm库

from utils.LuMoSDKClient import LusterFrame


class FpsTracker:
    def __init__(self, window_seconds=1.0):
        self.timestamps = collections.deque()
        self.window = window_seconds

    def update(self):
        now = time.time()
        self.timestamps.append(now)
        # 移除窗口外的时间戳
        while self.timestamps and (now - self.timestamps[0] > self.window):
            self.timestamps.popleft()

    def get_fps(self):
        n = len(self.timestamps)
        if n <= 1:
            return 0.0
        duration = self.timestamps[-1] - self.timestamps[0]
        return (n - 1) / duration if duration > 0 else 0.0


class FrameReceiver(QThread):
    """增强型网络接收器"""

    update_frame_signal = pyqtSignal(object)
    update_status_signal = pyqtSignal(str)
    update_imu_signal = pyqtSignal(object)
    update_cap_stats_signal = pyqtSignal(dict)

    def __init__(self, host, port, data_dir):
        super().__init__()
        self.host = host
        self.port = port
        self.running = False

        self.data_dir = data_dir

        self.is_recording = False
        self.name = None
        self.luster_fps_tracker = FpsTracker(window_seconds=1.0)
        self.num_keypoints = 0
        self.init_luster()
        self.init_decoder()
        self.init_socket()
        self.init_video()

    def run(self):
        self.running = True
        self.accept_connection_thread = threading.Thread(
            target=self.accept_connection, daemon=True
        )
        self.accept_connection_thread.start()
        self.decode_worker_thread = threading.Thread(
            target=self.decode_worker, daemon=True
        )
        self.decode_worker_thread.start()
        self.stats_udp_thread = threading.Thread(
            target=self.stats_udp_thread, daemon=True
        )
        self.stats_udp_thread.start()
        self.lf_receive_thread.start()
        self.start_lf_decoder()
        self.lf_save_thread.start()

    def start_recording(self, name):
        self.is_recording = True
        self.name = name

    def start_lf_decoder(self, work_num=4):
        for _ in range(work_num):
            t = threading.Thread(target=self._decode_worker)
            t.daemon = True
            t.start()
            self.decoder_pool.append(t)

    def _decode_worker(self):
        from queue import Empty

        while (
            not self.lf_raw_queue.empty() or self.running
        ):  # 即使已经停止，如果还有数据就要解码
            try:
                raw_frame = self.lf_raw_queue.get(timeout=0.5)
                lf_decode_result = self._lf_decoder(raw_frame)
                if len(lf_decode_result["markers"]) >= 0:
                    self.lf_decoded_queue.put(lf_decode_result)
                    self.decode_counter += 1
                    self.luster_fps_tracker.update()
                    self.lf_raw_queue.task_done()
            except Empty:
                continue

    def _lf_decoder(self, raw_frame) -> dict:
        result = {}
        frame_id = raw_frame.FrameId
        uCameraSyncTime = raw_frame.uCameraSyncTime
        result["frame_id"] = frame_id
        result["uCameraSyncTime"] = uCameraSyncTime / 1_000_000

        result["markers"] = []
        result["rigids"] = []
        result["skeletons"] = []
        result["markersets"] = []
        for marker in raw_frame.markers:
            id = marker.Id
            Name = marker.Name
            X = marker.X
            Y = marker.Y
            Z = marker.Z
            single_marker = {}
            single_marker["marker_id"] = id
            single_marker["marker_name"] = Name
            single_marker["marker_x"] = X
            single_marker["marker_y"] = Y
            single_marker["marker_z"] = Z
            result["markers"].append(single_marker)

        for rigid in raw_frame.rigidBodys:
            # if rigid.IsTrack is True:
            id = rigid.Id
            Name = rigid.Name
            X = rigid.X
            Y = rigid.Y
            Z = rigid.Z
            qx = rigid.qx
            qy = rigid.qy
            qz = rigid.qz
            qw = rigid.qw
            QualityGrade = rigid.QualityGrade
            single_rigid = {}
            single_rigid["rigid_id"] = id
            single_rigid["rigid_name"] = Name
            single_rigid["rigid_x"] = X
            single_rigid["rigid_y"] = Y
            single_rigid["rigid_z"] = Z
            single_rigid["rigid_qx"] = qx
            single_rigid["rigid_qy"] = qy
            single_rigid["rigid_qz"] = qz
            single_rigid["rigid_qw"] = qw
            single_rigid["QualityGrade"] = QualityGrade
            result["rigids"].append(single_rigid)

        for skeleton in raw_frame.customSkeleton:
            # if skeleton.IsTrack is True:
            id = skeleton.Id
            Name = skeleton.Name
            Type = skeleton.Type
            single_skeleton = {}
            single_skeleton["skeleton_id"] = id
            single_skeleton["skeleton_name"] = Name
            single_skeleton["skeleton_type"] = Type
            single_skeleton["keypoints"] = []
            for joint in skeleton.customSkeletonBones:
                id = joint.Id
                Name = joint.Name
                X = joint.X
                Y = joint.Y
                Z = joint.Z
                qx = joint.qx
                qy = joint.qy
                qz = joint.qz
                qw = joint.qw
                single_joint = {}
                single_joint["joint_id"] = id
                single_joint["joint_name"] = Name
                single_joint["joint_x"] = X
                single_joint["joint_y"] = Y
                single_joint["joint_z"] = Z
                single_joint["joint_qx"] = qx
                single_joint["joint_qy"] = qy
                single_joint["joint_qz"] = qz
                single_joint["joint_qw"] = qw
                single_skeleton["keypoints"].append(single_joint)
            result["skeletons"].append(single_skeleton)

        for markerset in raw_frame.markerSet:
            markerset_name = markerset.Name
            single_markerset = {}
            single_markerset["markerset_name"] = markerset_name
            single_markerset["markers"] = []
            for marker in markerset.markers:
                id = marker.Id
                Name = marker.Name
                X = marker.X
                Y = marker.Y
                Z = marker.Z
                single_marker = {}
                single_marker["marker_id"] = id
                single_marker["marker_name"] = Name
                single_marker["marker_x"] = X
                single_marker["marker_y"] = Y
                single_marker["marker_z"] = Z
                single_markerset["markers"].append(single_marker)

        # self.num_keypoints = len(result["skeletons"][0]["keypoints"])
        if len(result["skeletons"]) > 0:
            self.num_keypoints = len(result["skeletons"][0]["keypoints"])
        else:
            self.num_keypoints = 0
        return result

    def decode_worker(self):
        dctx = zstandard.ZstdDecompressor()
        fps_tracker = FpsTracker(window_seconds=1.0)
        while self.running:
            try:
                payload = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                data = pickle.loads(payload)
                if not isinstance(data, dict):
                    raise ValueError("无效数据格式")
                if data.get("type") != "frame":
                    continue  # 非视频数据，跳过

                compressed = data["frame"]
                timestamp = data.get("timestamp", 0)

                jpeg_data = dctx.decompress(compressed)
                frame = cv2.imdecode(
                    np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR
                )
                if frame is None:
                    raise ValueError("JPEG 解码失败")

                self.update_frame_signal.emit(frame)
                fps_tracker.update()

                self.last_report = time.time()

            except Exception as e:
                logging.warning(f"解码异常: {e}")
                self.send_control({"action": "reset"})
        print("解码线程结束")

    def send_control(self, command):
        try:
            data = json.dumps(command).encode()
            header = struct.pack(">I", len(data))
            self.control_conn.sendall(header + data)
        except Exception as e:
            print(f"控制指令发送失败: {str(e)}")

    def receive_imu_data(self):
        while self.running:
            try:
                data, addr = self.imu_socket.recvfrom(1024)  # 接收数据
                # print(f"接收到IMU数据: {data}")
                if len(data) == struct.calcsize("!ffffffffff"):
                    # 解析数据，与客户端打包格式一致
                    (
                        accel_x,
                        accel_y,
                        accel_z,
                        gyro_x,
                        gyro_y,
                        gyro_z,
                        mag_x,
                        mag_y,
                        mag_z,
                        timestamp,
                    ) = struct.unpack("!ffffffffff", data)
                    # print(struct.unpack('!ffffffffff', data))
                    # 处理IMU数据（示例：打印或存储）
                    # print(timestamp)
                    timestamp = time.time()
                    # print(f"IMU数据: {timestamp}, {accel_x}, {accel_y}, {accel_z}")
                    self.update_imu_signal.emit(
                        {
                            "timestamp": timestamp,
                            "accel": [accel_x, accel_y, accel_z],
                            "gyro": [gyro_x, gyro_y, gyro_z],
                            "mag": [mag_x, mag_y, mag_z],
                        }
                    )
            except socket.timeout:
                continue
            except Exception as e:
                print(f"接收IMU数据异常: {e}")

    def accept_connection(self):
        self.control_conn = None
        self.video_conn = None

        self.video_socket.settimeout(1.0)
        self.control_socket.settimeout(1.0)

        while self.running:
            try:
                # 接受视频连接
                if self.video_conn is None:
                    video_conn, addr = self.video_socket.accept()
                    self.video_conn = video_conn
                    self.update_status_signal.emit(f"视频通道连接: {addr}")
                    logging.info(f"视频通道连接: {addr}")

                # 接受控制连接
                if self.control_conn is None:
                    control_conn, addr = self.control_socket.accept()
                    self.control_conn = control_conn
                    self.update_status_signal.emit(f"控制通道连接: {addr}")
                    logging.info(f"控制通道连接: {addr}")

                # 连接成功后处理数据接收（可能多次循环重连）
                if self.video_conn and self.control_conn:
                    try:
                        self.handle_data_stream()
                    except Exception as e:
                        self.update_status_signal.emit(f"视频处理异常: {e}")
                        logging.warning(f"视频处理异常: {e}")
                    finally:
                        # ⚠️ 释放连接（以便下一次accept）
                        if self.video_conn:
                            try:
                                self.video_conn.close()
                            except:
                                pass
                            self.video_conn = None

            except socket.timeout:
                continue
            except Exception as e:
                self.update_status_signal.emit(f"连接错误: {e}")
                logging.error(f"连接错误: {e}")
                time.sleep(1)

    def cleanup_connection(self, *conns):
        print("正在关闭连接...")
        for conn in conns:
            if conn:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except Exception as e:
                    print(f"关闭连接时出错: {str(e)}")
                finally:
                    conn.close()

    def _recv_all(self, conn, nbytes, soft_timeout=10.0):
        buf = bytearray()
        start = time.monotonic()
        conn.settimeout(None)  # 阻塞模式
        while len(buf) < nbytes:
            if time.monotonic() - start > soft_timeout:
                raise TimeoutError("recv timeout")
            chunk = conn.recv(min(65536, nbytes - len(buf)))
            buf.extend(chunk)
        return bytes(buf)

    def handle_data_stream(self):
        conn = self.video_conn
        try:
            while self.running:
                try:
                    # 1) 读取长度头
                    header = self._recv_all(conn, 4)
                    msg_size = struct.unpack(">L", header)[0]
                    if msg_size > 50_000_000:
                        logging.warning(f"跳过超大消息: {msg_size} bytes")
                        self._recv_all(conn, msg_size)  # 丢掉这帧
                        continue

                    # 2) 读取 payload
                    payload = self._recv_all(conn, msg_size)

                    # 3) 推入队列
                    try:
                        self.frame_queue.put(payload, block=False)
                    except queue.Full:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(payload, block=False)

                except (TimeoutError, ConnectionResetError) as e:
                    logging.warning(f"接收中断: {e}")
                    break

        except Exception as e:
            self.update_status_signal.emit(f"视频接收异常: {e}")

    def stats_udp_thread(self):
        """从状态UDP通道接收并直接解码统计信息"""
        sock = self.status_socket
        sock.setblocking(False)

        while self.running:
            try:
                data, _ = sock.recvfrom(8192)
                if len(data) < 4:
                    continue

                msg_size = struct.unpack(">L", data[:4])[0]
                payload = data[4:]
                if msg_size != len(payload):
                    continue

                stats = pickle.loads(payload)
                if stats.get("type") == "stats":
                    stats["keypoints_fps"] = self.luster_fps_tracker.get_fps()
                    stats["num_keypoints"] = self.num_keypoints
                    self.update_cap_stats_signal.emit(stats)

            except BlockingIOError:
                time.sleep(0.05)
            except (pickle.UnpicklingError, ValueError) as e:
                logging.warning(f"状态包解码失败: {e}")
            except Exception as e:
                logging.warning(f"状态接收异常: {e}")

    def init_luster(self):
        self.lf_raw_queue = Queue(maxsize=300)
        self.lf_decoded_queue = Queue(maxsize=300)
        self.lf_receive_thread = threading.Thread(target=self.receive_luster_data)
        self.lf_save_thread = threading.Thread(
            target=self.lf_saver_thread, args=(1.0, "data_stream.jsonl")
        )  # TODO: set the proper batch size
        self.decoder_pool = []

        self.receive_counter = 0
        self.decode_counter = 0
        self.save_counter = 0
        self.batch_count = 0
        self.paused = False
        self.temp_dir = os.path.join(self.data_dir, "temp")

    def init_decoder(self):
        pass

    def init_socket(self):
        self.video_socket = self.create_tcp_socket(self.port)
        self.status_socket = self.create_udp_socket(self.port + 1)  # e.g., 9001
        self.control_socket = self.create_tcp_socket(self.port + 2)  # e.g., 9002

    def init_video(self):
        self.frame_queue = Queue(maxsize=1)
        self.stats_queue = Queue(maxsize=30)
        self.stat_window = Queue(maxsize=30)
        self.last_report = time.monotonic()

    def create_udp_socket(self, port):
        """创建并绑定一个 UDP socket。"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        print(f"📡 UDP socket 绑定成功：0.0.0.0:{port}")
        return sock

    def create_tcp_socket(self, port, backlog=1):
        """创建并监听一个 TCP socket，用于服务端等待连接。"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.listen(backlog)
        print(f"🎮 TCP socket 正在监听：0.0.0.0:{port}")
        return sock

    def receive_luster_data(self):
        ip = "127.0.0.1"
        self.lf = LusterFrame()
        self.lf.Connnect(ip)
        print(f"连接Luster数据: {ip}")
        while self.running:
            try:
                frame = self.lf.ReceiveData(flag=1)
                if frame:
                    self.lf_raw_queue.put(frame, timeout=0.1)
            except Exception as e:
                print(f"接收Luster数据异常: {e}")
                self.recover_connection()

    def recover_connection(self):
        self.lf.Close()
        self.lf = LusterFrame()
        self.lf.Connnect("127.0.0.1")

    def take_photo(self):
        # 发送保存图片命令
        self.send_control({"action": "take_photo"})

    def lf_saver_thread(self, flush_timeout=1.0, jsonl_filename="data_stream.jsonl"):
        """
        将解码后的数据流以 JSON Lines 格式实时写入文件，替代原有批量写入
        """
        from queue import Empty

        os.makedirs(self.temp_dir, exist_ok=True)
        file_path = os.path.join(self.temp_dir, jsonl_filename)
        # 以追加模式打开，一旦程序重启可继续写入
        with open(file_path, "a", encoding="utf-8") as f:
            while not self.lf_decoded_queue.empty() or self.running:
                try:
                    data = self.lf_decoded_queue.get(timeout=0.1)
                    # 写入一行 JSON 对象
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
                    f.flush()
                    self.save_counter += 1
                    self.lf_decoded_queue.task_done()
                except Empty:
                    # 队列空时等待后继续
                    time.sleep(min(flush_timeout, 0.1))

    def merge_files(
        self, jsonl_filename="data_stream.jsonl", final_filename="merged.json"
    ):
        """
        将 JSONL 文件合并为一个 JSON 数组文件
        """
        jsonl_path = os.path.join(self.temp_dir, jsonl_filename)
        final_path = final_filename
        with open(jsonl_path, "r", encoding="utf-8") as f_in, open(
            final_path, "w", encoding="utf-8"
        ) as f_out:
            f_out.write("[\n")
            first = True
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                if not first:
                    f_out.write(",\n")
                f_out.write(line)
                first = False
            f_out.write("\n]")
        print(f"Merged JSONL into {final_path}")

    def cleanup(self, jsonl_filename="data_stream.jsonl", retries=5, retry_delay=0.1):
        """
        清理流写入过程中产生的 JSONL 临时文件，带重试以处理文件占用问题
        """
        path = os.path.join(self.temp_dir, jsonl_filename)
        if os.path.exists(path):
            for attempt in range(1, retries + 1):
                try:
                    os.remove(path)
                    print(f"Removed {path}")
                    break
                except PermissionError:
                    if attempt < retries:
                        time.sleep(retry_delay)
                    else:
                        print(
                            f"Failed to remove {path} after {retries} retries: file may still be in use."
                        )
            # 重置计数器
            self.save_counter = 0
            return
        print(f"No JSONL file to clean at {path}")

    def stop(self, command):
        # with QMutexLocker(self.mutex):
        self.running = False
        self.send_control(command)
        if command["reply"] == True:
            self.start_file_receiver(save_dir=self.data_dir)
            keypoints_file = os.path.join(
                self.data_dir, self.name, "raw_keypoints.json"
            )
            self.merge_files(
                jsonl_filename="data_stream.jsonl",
                final_filename=keypoints_file,
            )
            self.cleanup(jsonl_filename="data_stream.jsonl")

    def start_file_receiver(self, save_dir="./received_files", ip="0.0.0.0", port=5001):
        """在子线程中启动接收函数"""
        thread = threading.Thread(
            target=self._file_receiver_once, args=(save_dir, ip, port), daemon=True
        )
        thread.start()

    def _file_receiver_once(self, save_dir, ip, port):
        os.makedirs(save_dir, exist_ok=True)
        temp_zip_path = os.path.join(save_dir, "temp_received.zip")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.bind((ip, port))
            server_sock.listen(1)
            print(f"[FileReceiver] 监听 {ip}:{port} ... 等待连接")

            try:
                conn, addr = server_sock.accept()
                print(f"[FileReceiver] 接收到连接来自 {addr}")
                with conn:
                    file_size_packed = conn.recv(8)
                    if len(file_size_packed) < 8:
                        raise ValueError("未正确接收到文件大小信息")
                    file_size = struct.unpack("!Q", file_size_packed)[0]
                    print(f"[FileReceiver] 文件大小：{file_size} 字节")

                    received_bytes = 0
                    with open(temp_zip_path, "wb") as f:
                        while received_bytes < file_size:
                            chunk = conn.recv(4096)
                            if not chunk:
                                break
                            f.write(chunk)
                            received_bytes += len(chunk)
                            percent = (received_bytes / file_size) * 100
                            print(
                                f"\r[FileReceiver] 接收进度: {received_bytes}/{file_size} ({percent:.2f}%)",
                                end="",
                            )

                    print("\n[FileReceiver] 接收完成：", temp_zip_path)

                    # 解压
                    extract_dir = os.path.join(save_dir, self.name)
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                    print(f"[FileReceiver] 解压完成：{extract_dir}")

            except Exception as e:
                logging.exception(f"[FileReceiver] 接收失败: {e}")

            finally:
                if os.path.exists(temp_zip_path):
                    try:
                        os.remove(temp_zip_path)
                        print(f"[FileReceiver] 删除临时文件：{temp_zip_path}")
                    except Exception as e:
                        logging.warning(f"[FileReceiver] 清理失败: {e}")

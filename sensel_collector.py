#sensel采集进程、按照固定fps采集arr与contact数据，与views进程通信

import os
import sys
import ctypes
import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import queue
import cv2
import csv
from multiprocessing import Process, Queue
import os, sys

# 1. 动态定位 wrapper 目录  
here = os.path.dirname(__file__)  
wrapper_dir = os.path.normpath(os.path.join(  
    here,  
    'sensel-api-master',
    'sensel-lib-wrappers',  
    'sensel-lib-python'  
))  
# 2. 检查目录是否存在  
if not os.path.isdir(wrapper_dir):  
    raise FileNotFoundError(f"找不到 Sensel wrapper 目录: {wrapper_dir}")  
# 3. 插到 sys.path 最前面  
if wrapper_dir not in sys.path:  
    sys.path.insert(0, wrapper_dir)  

# 4. 现在就可以 import sensel 了  
import sensel



class DummyFpsTracker:
    def __init__(self):
        self.last_time = time.time()
        self.frame_count = 0

    def update(self):
        self.frame_count += 1

    def get_fps(self):
        elapsed = time.time() - self.last_time
        if elapsed == 0:
            return 0
        fps = self.frame_count / elapsed
        self.last_time = time.time()
        self.frame_count = 0
        return fps

class SenselCollector:
    """_summary_
    Sensel 采集器，负责采集 Sensel 数据并写入文件
    外部通信接口：
    - start_recording()：开始录制
    - stop_recording()：停止录制
    - sensel_listener()：监听主界面的控制，开始，结束，暂停，恢复
    - send_sensel_thread()：以 30Hz 的频率发送数据
    """
    def __init__(self, 
                 cmd_queue: Queue,#cmd命令的进程间通信队列
                 data_send_queue:Queue,#压力图数据发送队列
                 state_send_queue:Queue,#状态数据发送队列
                 capture_fps=200,
                 send_fps=60):
        self.capture_fps = capture_fps
        self.send_fps = send_fps
        #定义写入文件路径
        
        self.root='sensel_data'
        #没有则创建
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        #定义写入的文件夹名，从监听线程中拿
        self.write_dir=None
        self.is_recording= False#初始化时为False
        #进程自己的线程写入队列
        self.write_frame_queue = queue.Queue(maxsize=3)
        self.send_frame_queue = queue.Queue(maxsize=3)
        #进程间的发送队列
        self.data_send_queue = data_send_queue
        self.cmd_queue = cmd_queue
        self.state_send_queue = state_send_queue
        self.is_recording = False  # 录制开关
        self.paused = False
        #定义capture fps统计
        self.cap_fps_tool = DummyFpsTracker()
        self.send_fps_tool = DummyFpsTracker()
        self.save_fps_tool = DummyFpsTracker()
        self.last_report = time.time()
        self.sensel_send_fps = None
        self.init_device()
        self.running = threading.Event()
        self.running.set()
        #初始化时开始线程
        #self.run()

    def init_device(self):
        self.handle = self.open_sensel()

        error, info = sensel.getSensorInfo(self.handle)
        if error != 0:
            raise RuntimeError("Failed to get sensor info")
        
        self.frame = self.init_frame(self.handle)
        self.width_mm, self.height_mm = info.width, info.height
        #print(f"Sensel device initialized: {self.width_mm} x {self.height_mm} mm")
        self.rows, self.cols = info.num_rows, info.num_cols
        # 1) 预先构造 ctypes 数组类型，加速 cast
        self._BufType = ctypes.c_float * (self.rows * self.cols)
        # 2) 预分配一个 numpy 缓冲区，后面直接 copyto
        self._arr_buffer = np.empty((self.rows, self.cols), dtype=np.float32)
        
    def open_sensel(self):
        error, dev_list = sensel.getDeviceList()
        if error != 0 or dev_list.num_devices == 0:
            raise RuntimeError("No Sensel device found")
        error, handle = sensel.openDeviceByID(dev_list.devices[0].idx)
        #sensel.sensel_lib.senselSetContactParams(handle, c_float(0.2), c_int(1))
        return handle

    def init_frame(self,handle):
        # 同时订阅压力图和触点信息
        mask = sensel.FRAME_CONTENT_PRESSURE_MASK | sensel.FRAME_CONTENT_CONTACTS_MASK
        sensel.setFrameContent(handle, mask)
        error, frame = sensel.allocateFrameData(handle)
        sensel.startScanning(handle)
        return frame

    def close_sensel(self,handle, frame):
        sensel.freeFrameData(handle, frame)
        sensel.stopScanning(handle)
        sensel.close(handle)    
    
    
    def run(self):
        self._threads = [
            threading.Thread(target=self.capture_sensel_thread, daemon=True),
            threading.Thread(target=self.write_sensel_thread, daemon=True),
            threading.Thread(target=self.cmd_listener, daemon=True),
            threading.Thread(target=self.send_state_thread, daemon=True),
            threading.Thread(target=self.send_sensel_thread, daemon=True),
        ]
        for t in self._threads:
            t.start()
        print("Sensel collector threads started.")
        self._send_thread_started = True
    
    
    def stop(self):
        self.running.clear()
        for t in self._threads:
            t.join()
        self.close_sensel(self.handle, self.frame)
        print("Sensel collector stopped.")   
         
    def capture_sensel_thread(self):
        """以 200 Hz 频率不断读取 Sensel 并入各个队列
            capture只负责采集原始数据
            write负责解码并写入文件
            """
        target_interval = 1.0 / self.capture_fps
        last_time = time.time()
        print("Sensel collector started.")
        while self.running.is_set():
            # —— 限速到 capture_fps —— 
            now = time.time()
            delta = now - last_time
            if delta < target_interval:
                time.sleep(target_interval - delta)
            last_time = time.time()
            # —— 读取一帧 Sensel 数据 —— 
            sensel.readSensor(self.handle)
            _, n = sensel.getNumAvailableFrames(self.handle)
            if n > 0:
                sensel.getFrame(self.handle, self.frame)
                self.cap_fps_tool.update()
                # 1) 快速 cast 到 ctypes buffer
                buf = ctypes.cast(self.frame.force_array,ctypes.POINTER(self._BufType)).contents
                # 2) 从 buffer 建立 numpy 视图并 reshape
                tmp = np.frombuffer(buf, dtype=np.float32).reshape(self.rows, self.cols)
                # 3) copy 到预分配的缓冲区（可选）
                np.copyto(self._arr_buffer, tmp)
                # 4) 返回副本，防止后续改写影响底层
                arr = self._arr_buffer.copy()
                #获取触点信息
                contacts = [
                            {"contact_id": c.id,
                            "x": c.x_pos,
                            "y": c.y_pos,
                            "force": c.total_force}
                            for c in self.frame.contacts[:self.frame.n_contacts]
                        ]
                #recording 时再给写线程入写文件队列
                if not self.paused:
                    ts = time.time()
                    try:
                        self.write_frame_queue.put((ts,arr,contacts), block=False)#直接把原码放入队列
                        self.send_frame_queue=self.write_frame_queue
                        
                    except queue.Full:
                        pass
        
    # def decode_sensel_frame(self, frame):   
    #     # ③ 直接用缓存类型快速 cast
    #     BufType = ctypes.c_float * (self.rows * self.cols)
    #     float_buf = ctypes.cast(frame.force_array,ctypes.POINTER(BufType)).contents
    #     arr = np.ctypeslib.as_array(float_buf).reshape((self.rows, self.cols))
    #     # 获取触点信息
    #     contacts = [
    #                 {"contact_id": c.id,
    #                  "x": c.x_pos,
    #                  "y": c.y_pos,
    #                  "force": c.total_force}
    #                 for c in self.frame.contacts[:frame.n_contacts]
    #             ]   
    #     contacts=self.supply_contact(arr,contacts)
    #     return arr, contacts
        
        
    def write_sensel_thread(self):
        #不断从队列中读取数据写入文件
        """专门将采集到的 sensel 数据写入本地文件"""
        while self.running.is_set() or not self.frame_queue.empty():
            if not self.is_recording:
                #循环等待，直到开始录制
                # 这里的 self.is_recording 是一个布尔值，表示是否正在录制
                time.sleep(0.1)
                continue
            try:
                timestamp,arr,contacts = self.write_frame_queue.get(timeout=0.1)
                contacts = self.supply_contact(arr, contacts)
                #arr, contacts = self.decode_sensel_frame(frame)
            except queue.Empty:
                continue
            
            if contacts:
                # 1. 写入 contacts.csv（每个触点一行）
                for c in contacts:
                    self.contacts_writer.writerow([
                        self.sensel_frame_id,
                        timestamp,
                        c["contact_id"],
                        c["x"],
                        c["y"],
                        c["force"]
                    ])
            else:
                # 如果没有触点，写入一行空行
                self.contacts_writer.writerow([
                    self.sensel_frame_id,
                    timestamp
                ])
            self.contacts_file.flush()
            
            # 2. 保存压力arr帧
            frame_file = os.path.join(
                self.sensel_frames_folder,
                f"{self.sensel_frame_id:06d}.npy"
            )
            np.save(frame_file, arr)
            
            
            self.sensel_frame_id += 1
            self.save_fps_tool.update()
            
        # 循环结束后，关闭 contacts.csv
        if hasattr(self, "contacts_file") and not self.contacts_file.closed:
            self.contacts_file.close()
            
            
    def send_sensel_thread(self):
        # 以 30FPS 的频率发送 Sensel 数据到 GUI
        import time, queue
        target_interval = 1.0 / self.send_fps
        last_time = time.time()
        while self.running.is_set():
            try:
                timestamp, arr,contacts = self.send_frame_queue.get(timeout=0.1)
                #arr, contacts = self.decode_sensel_frame(frame)
                contacts = self.supply_contact(arr, contacts)
            except queue.Empty:
                #print("send_sensel_thread: send_frame_queue is empty")
                continue
            # 限制到 send_fps
            now = time.time()
            delta = now - last_time
            if delta < target_interval:
                time.sleep(target_interval - delta)
            last_time = time.time()
            if not self.paused:
                # 发送压力图和触点数据到 GUI
                self.data_send_queue.put((timestamp, arr, contacts))
                print(f"发送压力图, timestamp: {timestamp:.2f}, arr shape: {arr.shape}, contacts: {len(contacts)}")
                self.send_fps_tool.update()
            
            
    def send_state_thread(self):
        #发送进程状态数据
        while self.running.is_set():
            stats = {
                "type": "stats",
                "cap_fps": self.cap_fps_tool.get_fps(),
                "send_fps": self.send_fps_tool.get_fps(),
                "save_fps": self.save_fps_tool.get_fps(),
                "timestamp": time.time()
            }
            time.sleep(1.0)
            # 发送状态数据
            if not self.paused:
                self.state_send_queue.put(stats)
            
            
    def supply_contact(self,arr,contacts):
        """
        补充因力值过小未被 sensel API 检测到的弱触点。
        - arr: 2D 压力矩阵 (rows x cols)
        - contacts: 已有的 contacts 列表
        返回新的 contacts 列表 (原列表 + 新补充)
        """
        fallback_thresh = 0.05   # 自定义小阈值 (N)
        # 二值化掩码
        mask = (arr > fallback_thresh).astype('uint8') * 255
        # 连通域分析
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # 记录已有像素坐标，避免重复
        existing_px = []
        for c in contacts:
            px = int(c['x'] / self.width_mm * self.cols)
            py = int(c['y'] / self.height_mm * self.rows)
            existing_px.append((px, py))
        if contacts:
            next_id = max((c.get("contact_id", 0) for c in contacts), default=0)+1
        else:   
            next_id = 0
        # 遍历每个连通域
        for lbl in range(1, n_labels):
            # 面积太小可忽略
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 100:
                continue
            # 质心 (x_pixel, y_pixel)
            cx, cy = centroids[lbl]
            ix, iy = int(cx), int(cy)
            # 跳过已有触点附近
            if any(abs(ix - ex) <= 10 and abs(iy - ey) <= 10 for ex, ey in existing_px):
                continue
            
            # 计算该连通域的合力 (3x3 邻域求和)
            y0, y1 = max(iy - 1, 0), min(iy + 2, self.rows)
            x0, x1 = max(ix - 1, 0), min(ix + 2, self.cols)
            force = float(arr[y0:y1, x0:x1].sum())
            # 转回物理坐标 (mm)
            x_mm = cx * self.width_mm / self.cols
            y_mm = cy * self.height_mm / self.rows
            # 添加到列表
            contacts.append({
                "contact_id": next_id,
                "x": x_mm,
                "y": y_mm,
                "force": force
            })
            next_id += 1
            print(f"补充触点: {next_id-1}, x: {x_mm:.2f}, y: {y_mm:.2f}, force: {force:.2f}")
        return contacts
        
        
        
       
    def cmd_listener(self):
        while self.running.is_set():
            if self.cmd_queue.empty():
                time.sleep(0.1)
                continue
            msg = self.cmd_queue.get()
            # 拆包：支持 (cmd, data) 或 纯 cmd 字符串
            if isinstance(msg, (tuple, list)) and len(msg) == 2:
                cmd, param = msg
            else:
                cmd, param = msg, None

            # “exit” 单独处理，退出采集
            if param == "exit":
                self.stop()
                break
            if cmd == "start":
                # 开始录制，设置文件夹名
                self.start_recording()
            if cmd == "stop":
                # 停止录制
                self.stop_recording()
            if cmd =="subject_info":
                self.write_dir = param
                #print(f"新 subject_info: {self.write_dir}")
            
        
    def start_recording(self):
        #开始记录，受到sensel_listener的控制
        # 创建文件夹
        write_path=os.path.join(self.root, self.write_dir)
        if not os.path.exists(write_path):
            os.makedirs(write_path)
         # 1.初始化 contacts.csv —— 
        contacts_csv = os.path.join(write_path, "contacts.csv")
        self.contacts_file = open(contacts_csv, "w", newline="")
        self.contacts_writer = csv.writer(self.contacts_file)
        # 写入表头
        self.contacts_writer.writerow(["frame_id", "timestamp", "contact_id", "x", "y", "force"])
        self.contacts_file.flush()
        
        #2.更新 sensel 数据保存目录
        self.sensel_frames_folder = os.path.join(write_path, "sensel_frames")
        os.makedirs(self.sensel_frames_folder, exist_ok=True)
        
        self.sensel_frame_id = 0
        self.is_recording = True
        
        
    def stop_recording(self):
        #停止记录，受到sensel_listener的控制
        # 关闭文件
        if hasattr(self, "contacts_file") and not self.contacts_file.closed:
            self.contacts_file.close()
        self.is_recording = False





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
                 arr_send_queue:Queue,#压力图数据发送队列
                 state_send_queue:Queue,#状态数据发送队列
                 capture_fps=200,
                 send_fps=30):
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
        self.sensel_write_queue = queue.Queue(maxsize=3)
        #进程间的发送队列
        self.sensel_send_queue = arr_send_queue
        self.cmd_queue = cmd_queue
        self.state_send_queue = state_send_queue
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
        self.run()

    def init_device(self):
        self.handle = self.open_sensel()

        error, info = sensel.getSensorInfo(self.handle)
        if error != 0:
            raise RuntimeError("Failed to get sensor info")
        
        self.frame = self.init_frame(self.handle)
        self.width_mm, self.height_mm = info.width, info.height
        self.rows, self.cols = info.num_rows, info.num_cols
        
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
        ]
        for t in self._threads:
            t.start()

        self._send_thread_started = True
    
    
    def stop(self):
        self.running.clear()
        for t in self._threads:
            t.join()
        self.close_sensel(self.handle, self.frame)
        print("Sensel collector stopped.")   
         
    def capture_sensel_thread(self):
        """以 200 Hz 频率不断读取 Sensel 并入各个队列"""
        target_interval = 1.0 / self.capture_fps
        last_time = time.time()

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
                # 每拿到一帧就 update()
                self.cap_fps_tool.update()
                sensel.getFrame(self.handle, self.frame)
                # 更新压力图
                BufType = ctypes.c_float * (self.rows * self.rows)
                float_buf = ctypes.cast(self.frame.force_array,
                                        ctypes.POINTER(BufType)).contents
                arr = np.ctypeslib.as_array(float_buf).reshape((self.rows, self.rows))
                # 获取触点信息并入队
                contacts = [
                    {"contact_id": c.id,
                     "x": c.x_pos,
                     "y": c.y_pos,
                     "force": c.total_force}
                    for c in self.frame.contacts[: self.frame.n_contacts]
                ]
                # 补充弱触点
                # 这里的 contacts 是一个列表，包含了所有触点的信息
                contacts=self.supply_contact(arr,contacts)

                # recording 时再给写线程入写文件队列
                if self.is_recording and not self.paused:
                    ts = time.time()
                    try:
                        self.sensel_write_queue.put((ts,arr,contacts),
                                                    block=False)
                        self.sensel_send_queue.put((ts, arr,contacts),
                                                    block=False)
                        
                    except queue.Full:
                        pass
        
        
    def write_sensel_thread(self):
        #不断从队列中读取数据写入文件
        """专门将采集到的 sensel 数据写入本地文件"""
        while self.running.is_set() or not self.sensel_write_queue.empty():
            if not self.is_recording:
                #循环等待，直到开始录制
                # 这里的 self.is_recording 是一个布尔值，表示是否正在录制
                time.sleep(0.1)
                continue
            try:
                timestamp,arr, contacts = self.sensel_write_queue.get(timeout=0.1)
            except queue.Empty:
                continue
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
            self.state_send_queue.put(stats)
            
            
    def supply_contact(self,arr,contacts):
        """
        补充因力值过小未被 sensel API 检测到的弱触点。
        - arr: 2D 压力矩阵 (rows x cols)
        - contacts: 已有的 contacts 列表
        返回新的 contacts 列表 (原列表 + 新补充)
        """
        fallback_thresh = 0.1   # 自定义小阈值 (N)
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
        next_id = max((c.get("contact_id", 0) for c in contacts), default=0) + 1
        # 遍历每个连通域
        for lbl in range(1, n_labels):
            # 面积太小可忽略
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < 2:
                continue
            # 质心 (x_pixel, y_pixel)
            cx, cy = centroids[lbl]
            ix, iy = int(cx), int(cy)
            # 跳过已有触点附近
            if any(abs(ix - ex) <= 1 and abs(iy - ey) <= 1 for ex, ey in existing_px):
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
        return contacts
        
        
        
       
    def cmd_listener(self):
        """监听 GUI 命令队列，支持带参数的命令 (cmd, param)"""
        while self.running.is_set():
            msg = self.cmd_queue.get()
            # 拆包：支持 ("start", "myname") 或 纯 "pause"/"resume" 字符串
            if isinstance(msg, (tuple, list)) and len(msg) == 2:
                cmd, param = msg
            else:
                cmd, param = msg, None

            if cmd == "start":
                # 从 GUI 拿到会话名，作为写入子目录
                self.write_dir = param or time.strftime("%Y%m%d_%H%M%S")
                self.start_recording()
            elif cmd == "stop":
                self.stop_recording()
            elif cmd == "pause":
                self.paused = True
            elif cmd == "resume":
                self.paused = False
            elif cmd == "exit":
                self.stop()
                break
        
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
        
        #   2.更新 sensel 数据保存目录
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
        
        
      
      
if __name__ == "__main__":
    # 测试 SenselCollector
    my_sensel = SenselCollector()
    pass
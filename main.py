import sys
import time
from multiprocessing import Manager, Process
from sensel_collector import SenselCollector
from model import FrameReceiver
from views import ControlView, GestureView
from controller import SystemController
from PyQt5.QtWidgets import QApplication
import subprocess

def collector_process(cmd_queue, data_queue, state_queue):
    sc = SenselCollector(
        cmd_queue=cmd_queue,
        data_send_queue=data_queue,
        state_send_queue=state_queue
    )
    sc.run()
    # 阻塞主线程，保持子进程存活，让后台线程持续运行
    try:
        while sc.running.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        sc.stop()
def call_remote_client_with_venv(user: str,
                                 host: str,
                                 remote_script: str,
                                 venv_activate_path: str):
    """
    user: 登录用户名
    host: 远程主机地址
    remote_script: client.py 在远程的绝对路径
    venv_activate_path: 远程虚拟环境 activate 脚本的绝对路径，
                        例如 /home/user/myenv/bin/activate
    """
    # 用 bash -lc 确保读到 .bashrc/.profile（必要时加载环境变量）
    remote_cmd = (
        f"bash -lc \"source {venv_activate_path} "
        f"&& python3 {remote_script}\""
    )
    ssh_cmd = ["ssh", f"{user}@{host}", remote_cmd]
    subprocess.Popen(ssh_cmd)
    
    
DATA_DIR = "data"
def main():
    mgr         = Manager()
    cmd_queue   = mgr.Queue(maxsize=3)
    data_queue  = mgr.Queue(maxsize=3)
    state_queue = mgr.Queue(maxsize=3)
    
    # 1) 启动 SenselCollector 子进程（不设为 daemon）
    p = Process(
        target=collector_process,
        args=(cmd_queue, data_queue, state_queue),
    )
    # 将子进程设为 daemon，主进程退出时自动终止
    p.daemon = True
    p.start()


    # #调用远程脚本
    # call_remote_client_with_venv(
    #                                 user="zhenqis123",
    #                                 host="183.173.107.7",
    #                                 remote_script="/home/zhenqis123/develop/test/client_azh.py",
    #                                 venv_activate_path="/home/zhenqis123/anaconda3/envs/wristband/bin/python"
    #                             )   
    
    # 2) 启动 Qt 主应用
    app = QApplication(sys.argv)

    # 3) 退出时通知子进程结束
    def on_about_to_quit():
        # 通知 SenselCollector 停止
        cmd_queue.put({"cmd": "exit"})
        # 等待子进程优雅退出
        p.join(timeout=2)
        # 如果仍在运行，强制终止
        if p.is_alive():
            p.terminate()
    app.aboutToQuit.connect(on_about_to_quit)

    # 4) MVC 初始化
    receiver = FrameReceiver(
        host="0.0.0.0", port=9999, data_dir=DATA_DIR,
        sensel_data_queue=data_queue,
        sensel_state_queue=state_queue
    )
    control_view = ControlView()
    gesture_view = GestureView()
    controller = SystemController(
        receiver, control_view, gesture_view,
        cmd_queue=cmd_queue,
        gesture_config_path="configs/gesture_config.json",
    )

    control_view.show()
    gesture_view.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
from multiprocessing import Manager, Process
from sensel_collector import SenselCollector
from model import FrameReceiver
from views import ControlView, GestureView
from controller import SystemController
import sys
from PyQt5.QtWidgets import QApplication

def main():
    # 1) 创建进程间队列
    mgr = Manager()
    cmd_queue = mgr.Queue(maxsize=3)
    data_queue = mgr.Queue(maxsize=3)
    stats_queue = mgr.Queue(maxsize=3)
    

    # 2) 启动 SenselCollector 进程,初始化三个进程共享队列
    def _collector_proc():
        sc = SenselCollector(cmd_queue=cmd_queue, data_queue=data_queue,stats_queue=stats_queue)
        sc.run()
    p = Process(target=_collector_proc, daemon=True)
    p.start()

    # 3) 启动 Qt 应用
    app = QApplication(sys.argv)
    receiver = FrameReceiver(data_queue=data_queue)
    gesture_view = GestureView()
    control_view = ControlView()

    # 4) GUI → collector: 发送命令
    control_view.control_event.connect(
        lambda evt, _: cmd_queue.put(evt)
    )
    # 5) collector → GUI: 更新 sensel 数据
    receiver.update_sensel_data_signal.connect(control_view.update_sensel_data)
    receiver.start()

    controller = SystemController(
        receiver, control_view, gesture_view,
        gesture_config_path="configs/gesture_config.json",
    )
    control_view.show()
    gesture_view.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
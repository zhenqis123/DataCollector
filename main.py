import sys
from PyQt5.QtWidgets import QApplication
from model import FrameReceiver
from views import ControlView, GestureView
from controller import SystemController
import multiprocessing

# import time

DATA_DIR = "data"


def main():
    app = QApplication(sys.argv)

    # 初始化模型
    receiver = FrameReceiver(host="0.0.0.0", port=9999, data_dir=DATA_DIR)

    # 初始化视图
    gesture_view = GestureView()
    control_view = ControlView()

    # 初始化控制器
    controller = SystemController(
        receiver,
        control_view,
        gesture_view,
        gesture_config_path="configs/gesture_config.json",
    )

    # 显示界面
    control_view.show()
    gesture_view.showMaximized()
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        print("🔴 用户中断了程序")
        sys.exit(0)


if __name__ == "__main__":
    main()

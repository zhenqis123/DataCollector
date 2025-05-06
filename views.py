from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QDialog,
    QSpacerItem,
)
from PyQt5.QtGui import QGuiApplication, QPalette, QColor, QFont, QIcon, QPixmap
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from PyQt5.QtMultimedia import QMediaPlayer, QMediaPlaylist, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl
import vlc


class ControlView(QWidget):
    control_event = pyqtSignal(str, object)  # Signal: event type and optional data
    imu_data_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        # Initialize data storage for sensor plots
        self.accel_data = {"x": [], "y": [], "z": [], "t": []}
        self.gyro_data = {"x": [], "y": [], "z": [], "t": []}
        self.setup_plots()
        self.init_ui()

        self.setStyleSheet(self._get_stylesheet())

    def init_ui(self):
        main_layout = QVBoxLayout()

        # ==== Participant Info Input Area ====
        subject_layout = QHBoxLayout()
        self.subject_input = QLineEdit()
        self.subject_input.setPlaceholderText("Enter participant name")
        btn_save_subject = QPushButton("Save Info")
        btn_save_subject.clicked.connect(self._handle_save_subject)
        self.subject_input.setStyleSheet(
            """
            QLineEdit {
                padding: 8px;
                border: 1px solid #546E7A;
                border-radius: 4px;
                min-width: 200px;
            }
        """
        )
        btn_save_subject.setStyleSheet(self._get_button_style())
        subject_layout.addWidget(QLabel("Participant Name:"))
        subject_layout.addWidget(self.subject_input)
        subject_layout.addWidget(btn_save_subject)
        subject_layout.addStretch()
        main_layout.addLayout(subject_layout)

        # ==== Video Display Area ====
        self.video_label = QLabel()
        self.video_label.setObjectName("video")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.video_label, stretch=4)

        # ==== Status Display Area (New) ====
        status_layout = QHBoxLayout()
        self.label_cap_fps = QLabel("Capture FPS: 0.0")
        self.label_save_fps = QLabel("Save FPS: 0.0")
        self.label_send_fps = QLabel("Send FPS: 0.0")
        self.label_keypoints_fps = QLabel("Keypoints FPS: 0.0")
        self.label_loss_rate = QLabel("Loss Rate: 0.0%")
        self.label_rtt = QLabel("RTT: 0 ms")
        self.label_keypoints_num = QLabel("Keypoints: 0")

        for label in [
            self.label_cap_fps,
            self.label_save_fps,
            self.label_send_fps,
            self.label_loss_rate,
            self.label_rtt,
            self.label_keypoints_fps,
            self.label_keypoints_num,
        ]:
            label.setStyleSheet(
                """
                QLabel {
                    padding: 8px;
                    background: #37474F;
                    color: #ECEFF1;
                    border-radius: 4px;
                }
            """
            )
            label.setAlignment(Qt.AlignCenter)
            status_layout.addWidget(label)

        main_layout.addLayout(status_layout)
        # ==== Charts Area (Acceleration & Gyroscope) ====
        charts_widget = QWidget()
        charts_layout = QHBoxLayout(charts_widget)
        charts_layout.addWidget(self.accel_canvas)
        charts_layout.addWidget(self.gyro_canvas)
        main_layout.addWidget(charts_widget, stretch=6)

        # ==== Control Buttons ====
        control_buttons = self._create_control_buttons()
        main_layout.addLayout(control_buttons)

        # ==== Progress Bar ====
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.setLayout(main_layout)
        self.setMinimumSize(1280, 800)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(
        self, cap_fps, save_fps, send_fps, loss_rate, rtt, keypoints_fps, keypoints_num
    ):
        self.label_cap_fps.setText(f"Capture FPS: {cap_fps:.2f}")
        self.label_save_fps.setText(f"Save FPS: {save_fps:.2f}")
        self.label_send_fps.setText(f"Send FPS: {send_fps:.2f}")
        self.label_keypoints_fps.setText(f"Keypoints FPS: {keypoints_fps:.2f}")
        self.label_loss_rate.setText(f"Loss Rate: {loss_rate:.2f}%")
        self.label_rtt.setText(f"RTT: {rtt:.2f} ms")
        self.label_keypoints_num.setText(f"Keypoints: {keypoints_num}")

    def _handle_save_subject(self):
        name = self.subject_input.text().strip()
        if name:
            self.control_event.emit("subject_info", name)
            self._show_save_success()
        else:
            self._show_save_warning()

    def _show_save_success(self):
        self.subject_input.setStyleSheet(
            """
            QLineEdit {
                border: 2px solid #4CAF50;
                background: #E8F5E9;
            }
        """
        )
        QTimer.singleShot(1500, lambda: self.subject_input.setStyleSheet(""))

    def _show_save_warning(self):
        self.subject_input.setStyleSheet(
            """
            QLineEdit {
                border: 2px solid #FF5252;
                background: #FFECB3;
            }
        """
        )
        QTimer.singleShot(1500, lambda: self.subject_input.setStyleSheet(""))

    def setup_plots(self):
        # ----- Acceleration Plot -----
        self.accel_fig = Figure(figsize=(6, 3))
        self.accel_ax = self.accel_fig.add_subplot(111)
        self.accel_ax.set_title("Acceleration", fontsize=10)
        self.accel_ax.set_xlabel("Time (s)", fontsize=8)
        self.accel_ax.set_ylabel("m/s²", fontsize=8)
        self.accel_ax.tick_params(labelsize=8)
        (self.accel_line_x,) = self.accel_ax.plot([], [], "r-", lw=1, label="X")
        (self.accel_line_y,) = self.accel_ax.plot([], [], "g-", lw=1, label="Y")
        (self.accel_line_z,) = self.accel_ax.plot([], [], "b-", lw=1, label="Z")
        self.accel_ax.legend(fontsize=8, loc="upper right")
        self.accel_ax.grid(True, alpha=0.5)
        self.accel_canvas = FigureCanvas(self.accel_fig)

        # ----- Gyroscope Plot -----
        self.gyro_fig = Figure(figsize=(6, 3))
        self.gyro_ax = self.gyro_fig.add_subplot(111)
        self.gyro_ax.set_title("Gyroscope", fontsize=10)
        self.gyro_ax.set_xlabel("Time (s)", fontsize=8)
        self.gyro_ax.set_ylabel("rad/s", fontsize=8)
        self.gyro_ax.tick_params(labelsize=8)
        (self.gyro_line_x,) = self.gyro_ax.plot([], [], "r-", lw=1, label="X")
        (self.gyro_line_y,) = self.gyro_ax.plot([], [], "g-", lw=1, label="Y")
        (self.gyro_line_z,) = self.gyro_ax.plot([], [], "b-", lw=1, label="Z")
        self.gyro_ax.legend(fontsize=8, loc="upper right")
        self.gyro_ax.grid(True, alpha=0.5)
        self.gyro_canvas = FigureCanvas(self.gyro_fig)

    def _get_button_style(self):
        return """
            QPushButton {
                background: #388E3C;
                color: white;
                padding: 6px 12px;
                min-width: 80px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: #43A047;
            }
        """

    def _create_control_buttons(self):
        self.btn_start = QPushButton("Start Recording")
        self.btn_stop = QPushButton("Stop Recording")
        self.btn_pause = QPushButton("Pause Recording")
        self.btn_resume = QPushButton("Resume Recording")
        self.btn_reset = QPushButton("Reset")
        # self.btn_take_photo = QPushButton("Take Photo")
        self.btn_start_gesture_update = QPushButton("Start Gesture Update")

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_stop)
        button_layout.addWidget(self.btn_pause)
        button_layout.addWidget(self.btn_resume)
        button_layout.addWidget(self.btn_reset)
        # button_layout.addWidget(self.btn_take_photo)
        button_layout.addWidget(self.btn_start_gesture_update)
        button_layout.addStretch()

        self.btn_start.clicked.connect(lambda: self.control_event.emit("start", None))
        self.btn_stop.clicked.connect(lambda: self.control_event.emit("stop", None))
        self.btn_pause.clicked.connect(lambda: self.control_event.emit("pause", None))
        self.btn_resume.clicked.connect(lambda: self.control_event.emit("resume", None))
        self.btn_reset.clicked.connect(lambda: self.control_event.emit("reset", None))
        self.btn_start_gesture_update.clicked.connect(
            lambda: self.control_event.emit("start_gesture_update", None)
        )

        return button_layout

    def update_video(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def update_imu_data(self, data):
        timestamp = float(data["timestamp"])
        accel = [float(v) for v in data["accel"]]
        gyro = [float(v) for v in data["gyro"]]
        max_points = 200
        for i, axis in enumerate(["x", "y", "z"]):
            self.accel_data[axis].append(accel[i])
            self.accel_data[axis] = self.accel_data[axis][-max_points:]
        self.accel_data["t"].append(timestamp)
        self.accel_data["t"] = self.accel_data["t"][-max_points:]
        for i, axis in enumerate(["x", "y", "z"]):
            self.gyro_data[axis].append(gyro[i])
            self.gyro_data[axis] = self.gyro_data[axis][-max_points:]
        self.gyro_data["t"].append(timestamp)
        self.gyro_data["t"] = self.gyro_data["t"][-max_points:]
        self.accel_line_x.set_data(self.accel_data["t"], self.accel_data["x"])
        self.accel_line_y.set_data(self.accel_data["t"], self.accel_data["y"])
        self.accel_line_z.set_data(self.accel_data["t"], self.accel_data["z"])
        self.accel_ax.relim()
        self.accel_ax.autoscale_view(scalex=True, scaley=True)
        if len(self.accel_data["t"]) > 1:
            time_window = 10
            min_t = max(self.accel_data["t"][-1] - time_window, self.accel_data["t"][0])
            self.accel_ax.set_xlim(min_t, min_t + time_window)
        self.gyro_line_x.set_data(self.gyro_data["t"], self.gyro_data["x"])
        self.gyro_line_y.set_data(self.gyro_data["t"], self.gyro_data["y"])
        self.gyro_line_z.set_data(self.gyro_data["t"], self.gyro_data["z"])
        self.gyro_ax.relim()
        self.gyro_ax.autoscale_view(scalex=True, scaley=True)
        if len(self.gyro_data["t"]) > 1:
            time_window = 10
            min_t = max(self.gyro_data["t"][-1] - time_window, self.gyro_data["t"][0])
            self.gyro_ax.set_xlim(min_t, min_t + time_window)
        self.accel_canvas.draw_idle()
        self.gyro_canvas.draw_idle()

    def _get_stylesheet(self):
        return """
            QLabel#video {
                border: 2px solid #4A5664;
                background: black;
            }
            QPushButton {
                background: #37474F;
                border: 1px solid #546E7A;
                padding: 8px;
                min-width: 120px;
                color: #ECEFF1;
            }
            QPushButton:hover {
                background: #455A64;
            }
            QPushButton#emergency {
                background: #B71C1C;
                color: white;
                font-weight: bold;
            }
        """


class GestureView(QWidget):
    video_end_signal = pyqtSignal()  # ✅ VLC 播放完触发的信号

    def __init__(self):
        super().__init__()
        self.is_active = False
        self.instance = vlc.Instance()

        # ✅ 使用 MediaListPlayer
        self.media_player = self.instance.media_player_new()
        self.media_list_player = self.instance.media_list_player_new()
        self.media_list_player.set_media_player(self.media_player)
        self.media_list_player.set_playback_mode(vlc.PlaybackMode.loop)

        self.current_video_path = None
        self.init_ui()

    def play_video(self, video_path):
        print("播放视频:", video_path)
        self.media_list_player.stop()  # ✅ 自动停止当前播放
        self.current_video_path = video_path

        # 创建新的播放列表
        media_list = self.instance.media_list_new([video_path])
        self.media_list_player.set_media_list(media_list)

        # 设置显示窗口
        self.current_video_label.repaint()
        self.current_video_label.show()
        self.media_player.set_hwnd(int(self.current_video_label.winId()))

        self.media_list_player.play()

    def init_ui(self):
        self.setMinimumSize(640, 480)
        root_layout = QVBoxLayout()
        gestures_layout = QHBoxLayout()
        gestures_layout.setSpacing(20)
        gestures_layout.setContentsMargins(20, 20, 20, 20)

        # Current gesture area
        current_layout = QVBoxLayout()
        self.current_video_label = QLabel("(Video Area)")
        self.current_video_label.setMinimumSize(400, 400)
        self.current_video_label.setStyleSheet("background-color: black")
        current_layout.addWidget(self.current_video_label, stretch=3)

        self.current_info_label = QLabel()
        self.current_info_label.setAlignment(Qt.AlignCenter)
        self.current_info_label.setStyleSheet(
            """
            font: bold 50px 'Microsoft YaHei';
            color: #FF5722;
            background: rgba(0, 0, 0, 150);
            padding: 10px;
            border-radius: 15px;
            """
        )
        current_layout.addWidget(self.current_info_label, stretch=1)

        # Next gesture area
        next_layout = QVBoxLayout()
        self.next_gesture_label = QLabel()
        self.next_gesture_label.setMinimumSize(200, 200)
        self.next_gesture_label.setAlignment(Qt.AlignCenter)
        self.next_gesture_label.setStyleSheet("background-color: white")
        next_layout.addWidget(self.next_gesture_label, stretch=2)

        self.next_name_label = QLabel()
        self.next_name_label.setAlignment(Qt.AlignCenter)
        self.next_name_label.setStyleSheet(
            """
            font: bold 55px 'Microsoft YaHei';
            color: #FF5722;
            background: rgba(255, 255, 255, 200);
            padding: 12px;
            border-radius: 15px;
            """
        )
        next_layout.addWidget(self.next_name_label, stretch=1)

        gestures_layout.addLayout(current_layout, stretch=3)
        gestures_layout.addLayout(next_layout, stretch=1)
        root_layout.addLayout(gestures_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        root_layout.addWidget(self.progress_bar)

        self.setLayout(root_layout)

    def update_progress(self, value: int):
        self.progress_bar.setValue(value)

    def show_gesture(self, current_gesture: dict, next_gesture: dict = None):

        video_path = current_gesture.get("video", None)

        if video_path:
            print("Video path:", video_path)
            print("begin video play")
            self.current_video_path = video_path  # ✅ 保存路径用于循环
            self.play_video(video_path)

        if next_gesture:
            next_pixmap = next_gesture.get("qpixmap", None)
            if next_pixmap:
                self.next_gesture_label.setPixmap(
                    next_pixmap.scaled(
                        self.next_gesture_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
        else:
            self.next_gesture_label.clear()
            self.next_name_label.setText("-")

    def update_remaining_time(self, current_text, next_text):
        self.current_info_label.setText(current_text)
        if next_text is None:
            return
        self.next_name_label.setText(next_text)

    def clear_current_video(self):
        self.media_list_player.stop()
        self.current_video_label.clear()
        self.current_video_label.setStyleSheet("background-color: black")

    def show_random_info(self, text):
        self.current_info_label.setText(text)
        self.clear_current_video()
        self.next_gesture_label.clear()
        self.next_name_label.setText("-")

    def hint_over(self):
        self.current_info_label.setText(
            "Data Collection Completed! Please remove the device."
        )
        # self.current_gesture_label.clear()
        self.next_gesture_label.clear()
        self.next_name_label.setText("-")

    def _clear_display(self):
        self.current_video_label.clear()
        self.next_gesture_label.clear()
        self.current_info_label.setText("Press Start to Begin")
        self.next_name_label.setText("-")

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def move_to_screen(self, screen_index=0):
        screens = QGuiApplication.screens()
        if not screens:
            print("⚠️ 没有检测到屏幕！")
            return

        if screen_index >= len(screens):
            print(
                f"⚠️ 你要求的屏幕编号 {screen_index} 超出了最大屏幕数 {len(screens)}，使用最后一个屏幕。"
            )
            screen_index = len(screens) - 1

        target_screen = screens[screen_index]
        self.setMinimumSize(640, 480)
        self.showFullScreen()  # 先 show

        # 此时 windowHandle 不再是 None
        handle = self.windowHandle()
        if handle:
            handle.setScreen(target_screen)
        else:
            print("⚠️ windowHandle 仍然是 None，可能窗口未正确 show")


class SessionBreakDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Session Break")
        self.setModal(True)
        self.setStyleSheet(
            """
            QDialog {
                background: #1A237E;
                padding: 20px;
            }
            QLabel {
                color: white;
                font: bold 18px;
            }
            QPushButton {
                background: #FF5722;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
            }
        """
        )
        layout = QVBoxLayout()
        self.info_label = QLabel(
            "Session Completed!\nClick OK to resume gesture update."
        )
        layout.addWidget(self.info_label)
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        layout.addWidget(ok_btn)
        self.setLayout(layout)


class WhetherSaveDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("请选择是否传输并保存当前数据")
        self.setFixedSize(400, 200)

        self.setStyleSheet(
            """
            QDialog {
                background: #1A237E;
                padding: 20px;
            }
            QLabel {
                color: white;
                font: bold 18px;
            }
            QPushButton {
                background: #FF5722;
                color: white;
                padding: 10px 20px;
                border-radius: 5px;
            }
            """
        )

        layout = QVBoxLayout()
        label = QLabel("请选择是否传输并保存当前数据")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        btn_ok = QPushButton("确定")
        btn_cancel = QPushButton("取消")

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def move_to_parent_center(self):
        if self.parent() is None:
            return
        parent_geom = self.parent().geometry()
        dialog_size = self.size()
        x = parent_geom.x() + (parent_geom.width() - dialog_size.width()) // 2
        y = parent_geom.y() + (parent_geom.height() - dialog_size.height()) // 2
        self.move(x, y)

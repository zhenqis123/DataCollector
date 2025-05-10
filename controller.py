import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
from controller_fsm import SystemStateMachine
from model import FrameReceiver
from PyQt5.QtCore import QObject, QTimer, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QDialog, QMessageBox
from transitions import Machine
from views import ControlView, GestureView, SessionBreakDialog, WhetherSaveDialog


class SystemController(QObject):
    def __init__(
        self,
        model: FrameReceiver,
        control_view: ControlView,
        gesture_view: GestureView,
        gesture_config_path: str = r"configs/gesture_config.json",
        data_dir: str = r"data",
    ):
        super().__init__()
        self.model = model
        self.data_dir = data_dir
        self.control_view = control_view
        self.gesture_view = gesture_view
        self.gesture_view.move_to_screen(1)
        self.fsm = SystemStateMachine(self)
        # self._wire_signals()
        self.gesture_config_path = gesture_config_path
        # Initialize subject information
        self.init_subject_info()
        # Initialize and preload gesture images
        self.init_gestures(self.gesture_config_path)
        # Setup connections (signals/slots)
        self.setup_connections()
        # Start the data receiver thread
        self.model.start()

    def init_subject_info(self):
        self.subject_info = {"name": "", "date": "", "data_dir": ""}
        self.subject_info["date"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def handle_subject_info(self, subject_data):
        """Update subject information."""
        self.subject_info["name"] = subject_data
        self.subject_info["data_dir"] = f"data/{subject_data}"
        print(
            f"Updated subject name: {subject_data}, Data Dir: {self.subject_info['data_dir']}"
        )

    def init_gestures(self, gesture_config_path):
        """Preload gesture images and set up session parameters."""

        def get_gesture_display_indexes(self):
            import random

            gesture_display_indexes = []
            indexes = [
                i
                for i in range(self.gesture_count)
                for _ in range(self.repeat_count_per_gesture)
            ]
            random.shuffle(indexes)
            gesture_display_indexes.extend(indexes)
            indexes = [
                random.randint(0, self.gesture_count - 1)
                for _ in range(self.random_gesture_count)
            ]
            random.shuffle(indexes)
            gesture_display_indexes.extend(indexes)
            return gesture_display_indexes

        self.gesture_image_qpixmaps = []
        with open(gesture_config_path, "r") as f:
            gesture_config = json.load(f)
        media_dir = gesture_config["media_dir"]
        media_dir = Path(media_dir)
        video_paths = sorted(media_dir.glob("*.mp4"), key=lambda x: x.stem)
        image_paths = sorted(media_dir.glob("*.jpg"), key=lambda x: x.stem)
        is_full = gesture_config.get("full", False)
        if is_full:
            self.session_count = gesture_config["session_count"]
            self.repeat_count_per_gesture = gesture_config["repeat_count_per_gesture"]
            self.random_gesture_count = gesture_config["random_gesture_count"]
            self.update_interval = gesture_config["update_interval"]
            self.random_gesture_session_duration = gesture_config[
                "random_gesture_session_time"
            ]
        else:
            video_paths = video_paths[:2]
            image_paths = image_paths[:2]
            self.session_count = 2
            self.repeat_count_per_gesture = 1
            self.random_gesture_count = 2
            self.update_interval = 3
            self.random_gesture_session_duration = 10
        video_paths, image_paths = zip(
            *[(v, i) for v, i in zip(video_paths, image_paths) if i.exists()]
        )
        video_paths = [str(v) for v in video_paths]
        image_paths = [str(i) for i in image_paths]
        self.gesture_image_qpixmaps = [
            QPixmap(image_path) for image_path in image_paths
        ]
        self.gesture_names = [
            os.path.splitext(os.path.basename(image_path))[0]
            for image_path in image_paths
        ]
        self.gesture_video_paths = video_paths

        self.gesture_count = len(self.gesture_image_qpixmaps)

        # 1. 整体生成一次展示索引
        full_display_indexes = get_gesture_display_indexes(self)
        self.gesture_display_indexes = full_display_indexes

        # 2. 均分到各 session，并计算 stop 索引
        n = len(full_display_indexes)
        per = n // self.session_count  # 每段基础长度
        self.session_stop_indexes = [(i + 1) * per for i in range(self.session_count)]

        # 3. 处理余数：如果有多余的索引，加到最后一轮
        if n % self.session_count != 0:
            self.session_stop_indexes[-1] = n
        self.current_gesture_count = 0
        self.current_gesture_index = 0
        self.current_session_index = 0

        self.time_left = float(self.update_interval)

        self.random_time_left = 0
        self.is_random_session = False

        self.gesture_queue = Queue()
        self.is_end = False
        # Timer for normal gesture update
        self.countdown_timer = QTimer()
        self.countdown_timer.timeout.connect(self.update_countdown)
        # Flag: gesture update starts only after user clicks the button.
        self.gesture_update_started = False

    def setup_timers(self):
        """Initialize timers if needed."""
        pass

    def update_countdown(self):
        if not self.is_active:
            return
        self.time_left -= 0.1
        if self.time_left <= 0:
            self.switch_to_next_gesture()
            self.time_left = self.update_interval
            self.update_display()
        self.update_remaining_time()

    def switch_to_next_gesture(self):
        self.current_gesture_count += 1
        if (
            self.current_gesture_count
            == self.session_stop_indexes[self.current_session_index]
        ):
            self.current_session_index += 1
            if self.current_session_index == len(self.session_stop_indexes):
                self.pause_gesture()
                self.start_random_session()
                return
            else:
                self.pause_gesture()
                return
        else:
            if not self.is_random_session:
                self.current_gesture_index = self.gesture_display_indexes[
                    self.current_gesture_count
                ]

    def update_display(self):
        if self.is_end:
            self.gesture_view.hint_over()
            return
        if self.is_random_session:
            self._show_random_session_display()
        elif not self.gesture_update_started and not self.is_random_session:
            # In pre-gesture phase, prompt user to click button to start gesture update.
            display_text = (
                "<span style='font-size:40pt; color:#FFFFFF;'>Recording in Progress, now is the calibration phase</span><br><br>"
                "<span style='font-size:30pt; color:#FFEB3B;'>When the gesture collection session begins, <br> please keep your gesture convertion slow</span><br>"
            )
            self.gesture_view.show_random_info(display_text)
        else:
            self._show_normal_display()

        total_gestures = len(self.gesture_display_indexes)
        progress = int((self.current_gesture_count + 1) / total_gestures * 100)
        progress = min(progress, 100)
        self.control_view.update_progress(progress)
        self.gesture_view.update_progress(progress)

    def update_remaining_time(self):
        if self.is_random_session:
            self._show_random_session_display()
        else:
            if self.gesture_image_qpixmaps and self.is_active:
                current_gesture_name = self.gesture_names[self.current_gesture_index]
                current_text = (
                    "Current Gesture: "
                    + current_gesture_name
                    + f"\nRemaining Time: {round(self.time_left, 2)} s"
                )
                if self.current_gesture_count <= len(self.gesture_display_indexes) - 2:
                    next_index = self.gesture_display_indexes[
                        self.current_gesture_count + 1
                    ]
                    next_text = "Next Gesture: " + self.gesture_names[next_index]
                else:
                    next_text = None
                self.gesture_view.update_remaining_time(current_text, next_text)

    def _show_random_session_display(self):
        display_text = (
            "<span style='font-size:40pt; color:#FFFFFF;'>Free Movement Phase</span><br><br>"
            f"<span style='font-size:30pt; color:#FFEB3B;'>Remaining Time: {round(self.random_time_left)} s</span><br>"
            "<span style='font-size:20pt; color:#B3E5FC;'>Maintain a natural slow pace and change gestures freely</span>"
        )
        self.gesture_view.show_random_info(display_text)

    def _show_normal_display(self):
        if self.gesture_image_qpixmaps and self.is_active:
            current_pixmap = self.gesture_image_qpixmaps[self.current_gesture_index]
            current_video_path = self.gesture_video_paths[self.current_gesture_index]
            current_gesture_name = self.gesture_names[self.current_gesture_index]
            current_text = (
                "Current Gesture: "
                + current_gesture_name
                + f"\nRemaining Time: {round(self.time_left, 2)} s"
            )
            current_gesture_info = {
                "text": current_text,
                "video": current_video_path,
                "qpixmap": current_pixmap,
            }
            if self.current_gesture_count <= len(self.gesture_display_indexes) - 2:
                next_index = self.gesture_display_indexes[
                    self.current_gesture_count + 1
                ]
                next_pixmap = self.gesture_image_qpixmaps[next_index]
                next_text = "Next Gesture: " + self.gesture_names[next_index]
                next_video_path = self.gesture_video_paths[next_index]

                next_gesture_info = {
                    "text": next_text,
                    "qpixmap": next_pixmap,
                    "video": next_video_path,
                }
                # print(current_gesture_info, next_gesture_info)
            else:
                next_gesture_info = None
            self.gesture_view.show_gesture(current_gesture_info, next_gesture_info)

    def update_random_countdown(self):
        self.random_time_left -= 0.1
        if self.random_time_left <= 0:
            self.end_random_session()
        self.update_display()

    def end_random_session(self):
        self.is_active = False
        self.is_random_session = False
        self.random_timer.stop()
        self.is_end = True

    def pause_gesture(self):
        self.is_active = False
        self.countdown_timer.stop()
        self.model.paused = True
        self.model.send_control({"action": "pause_recording"})
        # Display prompt on gesture view during pause
        self.gesture_view.show_random_info(
            "Recording paused. Awaiting confirmation to resume..."
        )
        dialog = SessionBreakDialog(self.control_view)
        dialog.accepted.connect(self.start_gesture)
        dialog.exec_()

    def start_random_session(self):
        self.countdown_timer.stop()
        self.is_random_session = True
        self.random_time_left = self.random_gesture_session_duration
        self.gesture_view._clear_display()
        self.random_timer = QTimer()
        self.random_timer.timeout.connect(self.update_random_countdown)
        self.random_timer.start(100)

    def _wire_signals(self):
        # Connect btns to slot
        self.control_view.control_event.connect(self.handle_control_event)

        # Connect signals to slots
        self.model.update_frame_signal.connect(self.handle_frame)
        self.model.update_status_signal.connect(self.handle_status)
        self.model.update_imu_signal.connect(self.handle_imu)
        self.model.update_cap_stats_signal.connect(self.handle_cap_stats)

    def setup_connections(self):
        self.model.update_frame_signal.connect(self.handle_frame)
        self.model.update_status_signal.connect(self.handle_status)
        self.model.update_imu_signal.connect(self.handle_imu)
        self.model.update_cap_stats_signal.connect(self.handle_cap_stats)
        self.control_view.control_event.connect(self.handle_control_event)

    @pyqtSlot(dict)
    def handle_cap_stats(self, cap_stats):
        cap_fps = cap_stats.get("cap_fps", 0)
        send_fps = cap_stats.get("send_fps", 0)
        save_fps = cap_stats.get("save_fps", 0)
        keypints_fps = cap_stats.get("keypoints_fps", 0)
        quality = cap_stats.get("quality", 0)
        loss_rate = cap_stats.get("loss_rate", 0)
        rtt = cap_stats.get("rtt", 0)
        keypoints_num = cap_stats.get("keypoints_num", 0)
        self.control_view.update_status(
            cap_fps, save_fps, send_fps, loss_rate, rtt, keypints_fps, keypoints_num
        )

    @pyqtSlot(np.ndarray)
    def handle_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.control_view.update_video(rgb_frame)

    @pyqtSlot(dict)
    def handle_imu(self, imu_data):
        self.control_view.update_imu_data(imu_data)

    @pyqtSlot(str)
    def handle_status(self, status):
        print(f"System Status: {status}")

    @pyqtSlot(str, object)
    def handle_control_event(self, event_type, data=None):
        event_handlers = {
            "start": (self.start_recording, False),
            "start_gesture_update": (self.start_gesture_update, False),
            "stop": (self.stop_recording, False),
            "pause": (self.pause_recording, False),
            "resume": (self.resume_recording, False),
            "reset": (self.reset, False),
            "take_photo": (self.take_photo, False),
            "subject_info": (self.handle_subject_info, True),
        }
        handler_info = event_handlers.get(event_type)
        if not handler_info:
            return
        handler, require_data = handler_info
        if require_data and not data:
            print(f"[Warning] {event_type} event requires data")
            return
        handler(data) if require_data else handler()

    def pause_recording(self):
        self.is_active = False
        self.model.send_control({"action": "pause_recording"})
        self.model.paused = True

    def resume_recording(self):
        self.is_active = True
        self.model.send_control({"action": "resume_recording"})
        self.model.paused = False

    def start_gesture(self):
        self.is_active = True
        self.model.paused = False
        self.model.send_control({"action": "resume_recording"})
        self.countdown_timer.start(100)
        self.gesture_view._clear_display()
        self.gesture_view.clear_current_video()
        self.update_display()

    def start_gesture_update(self):
        self.gesture_update_started = True
        self.start_gesture()

    def start_recording(self):
        if not self.subject_info.get("name", "").strip():
            QMessageBox.warning(
                self.control_view,
                "Warning",
                "Please fill in participant information!",
                QMessageBox.Ok,
            )
            return
        self.start_save()
        # Wait for user to click "Start Gesture Update" to begin gesture update phase.
        self.gesture_update_started = False

    def stop_recording(self):
        print("Collection complete, stopping recording")
        self.is_end = True
        self.gesture_view._clear_display()
        self.gesture_view.hint_over()
        self.stop_gesture()
        self.stop_save()
        sys.exit(0)

    def stop_gesture(self):
        self.is_active = False
        self.countdown_timer.stop()
        self.current_gesture_count = 0
        self.current_gesture_index = 0
        self.current_session_index = 0

    def take_photo(self):
        print("Photo taken successfully")
        self.model.take_photo()

    def start_save(self):
        print(f"Start recording: {self.subject_info['name']}")
        os.makedirs(
            os.path.join(self.data_dir, self.subject_info["name"]), exist_ok=True
        )
        basename = datetime.now().strftime(f"{self.subject_info['name']}_%Y%m%d_%H%M%S")
        video_filename = f"{basename}_video.avi"
        imu_filename = f"{basename}_imu.csv"
        video_path = os.path.join(
            self.data_dir, self.subject_info["name"], video_filename
        )
        imu_path = os.path.join(self.data_dir, self.subject_info["name"], imu_filename)
        self.subject_info["video_path"] = video_path
        self.subject_info["imu_path"] = imu_path
        command = {
            "action": "start_recording",
            "participant_info": self.subject_info,
        }
        self.model.send_control(command)
        self.model.start_recording(self.subject_info["name"])

    def stop_save(self):
        command = {"action": "stop_recording", "reply": True}
        dialog = WhetherSaveDialog(self.control_view)
        dialog.move_to_parent_center()
        result = dialog.exec_()
        if result == QDialog.Accepted:
            command["reply"] = True
        else:
            command["reply"] = False
        self.model.stop(command)

    def on_merge_finished(self):
        print("Merge finished")

    def reset(self):
        self.init_subject_info()
        self.init_gestures(self.gesture_config_path)

    def emergency_stop(self):
        self.model.stop()
        self.countdown_timer.stop()


from PyQt5.QtCore import QThread, pyqtSignal


class MergeFilesThread(QThread):
    finished = pyqtSignal()

    def __init__(self, model, keypoint_file_name):
        super().__init__()
        self.model = model
        self.keypoint_file_name = keypoint_file_name

    def run(self):
        self.model.merge_files(self.keypoint_file_name)
        self.finished.emit()

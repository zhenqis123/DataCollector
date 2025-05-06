# -------- FSM 设计 --------
from transitions import Machine
from PyQt5.QtCore import QTimer
from typing import TYPE_CHECKING


class SystemStateMachine:
    """
    Session-level Finite-State-Machine

    states:
        idle            – 等待用户点击“开始”
        prepare         – 倒计时 3-2-1
        gesture         – 播放普通手势
        break_session   – 休息 / 补水
        random          – 播放随机手势
        finished        – 全流程结束

    triggers:
        start()         – idle → prepare
        prepared()      – prepare → gesture
        gesture_done()  – gesture → break_session
        resume()        – break_session → random
        random_done()   – random → finished
        reset()         – 任意 → idle
    """

    def __init__(self, controller: "SystemController"):
        self.controller = controller

        self.machine = Machine(
            model=self,
            states=[
                "idle",
                "prepare",
                "gesture",
                "break_session",
                "random",
                "finished",
            ],
            initial="idle",
            send_event=True,
            queued=True,  # 串行处理触发
            auto_transitions=False,
        )

        # --- transitions ---
        self.machine.add_transition("start", "idle", "prepare", after="on_prepare")
        self.machine.add_transition(
            "prepared", "prepare", "gesture", after="on_gesture"
        )
        self.machine.add_transition(
            "gesture_done", "gesture", "break_session", after="on_break"
        )
        self.machine.add_transition(
            "resume", "break_session", "random", after="on_random"
        )
        self.machine.add_transition(
            "random_done", "random", "finished", after="on_finished"
        )
        self.machine.add_transition("reset", "*", "idle", after="on_reset")

    # ---------- state callbacks ----------
    def _countdown(self, secs: int, callback):
        """通用倒计时工具 – 在主 GUI 线程启动 QTimer"""
        self._remaining = secs
        self.controller.view_control.update_countdown(self._remaining)
        self._timer = QTimer()
        self._timer.timeout.connect(lambda: self._tick(callback))
        self._timer.start(1000)

    def _tick(self, callback):
        self._remaining -= 1
        self.controller.view_control.update_countdown(self._remaining)
        if self._remaining <= 0:
            self._timer.stop()
            callback()

    # ---- on_enter helpers ----
    def on_prepare(self, *_):
        # 3 秒倒计时后触发 prepared()
        self._countdown(3, self.prepared)

    def on_gesture(self, *_):
        self.controller.start_gesture_session()

    def on_break(self, *_):
        self.controller.pause_for_break()

    def on_random(self, *_):
        self.controller.start_random_session()

    def on_finished(self, *_):
        self.controller.finish_session()

    def on_reset(self, *_):
        self.controller.reset_session()

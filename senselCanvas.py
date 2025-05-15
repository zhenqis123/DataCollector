import numpy as np
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QImage, QPainter, QPen, QFont
from PyQt5.QtWidgets import QWidget
from matplotlib import cm

class SenselCanvas(QWidget):
    """原生 Qt 绘制 Sensel 压力图 & 触点"""
    def __init__(self, parent=None,
                 width_mm=230, height_mm=130):
        super().__init__(parent)
        self.arr = None
        self.contacts = []
        # 传感器物理尺寸，用于把 mm 映射到像素
        self.width_mm = width_mm
        self.height_mm = height_mm

    def set_frame(self, arr: np.ndarray, contacts: list):
        """外部调用传入新帧并刷新"""
        self.arr = arr
        self.contacts = contacts
        self.update()     # 触发 repaint

    def paintEvent(self, event):
        if self.arr is None:
            return
        h, w = self.arr.shape
        # 1) 归一化到 [0,1]
        min=0
        max=30
        norm = (self.arr - min) / (max + 1e-6)
        # 2) 应用 jet 伪彩色
        rgba = cm.jet(norm)                # (h, w, 4) float in [0,1]
        rgb = (rgba[..., :3] * 255).astype('uint8')  # 丢弃 alpha
        # 3) 转为 QImage (RGB888)
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # 3) 用 QPainter 绘制
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        # 拉伸填充整个控件
        painter.drawImage(self.rect(), qimg)

        # 4) 仅保留触点力值显示
        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        # 坐标映射 mm → 像素
        sx = self.width() / self.width_mm
        sy = self.height() / self.height_mm
        for c in self.contacts:
            x = c["x"] * sx
            y = c["y"] * sy
            text = f"{c['force']:.1f}"
            painter.drawText(QPointF(x, y), text)

        painter.end()
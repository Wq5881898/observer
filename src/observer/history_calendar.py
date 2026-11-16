"""Compact half-year activity calendar for the history page."""
from datetime import date, timedelta

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget, QLabel, QToolButton, QVBoxLayout, QHBoxLayout, QGridLayout, QSizePolicy, QLayout,
)

from observer.history import heat_level, minutes_text

COLORS = ('#eef0f3', '#dbeafe', '#93c5fd', '#60a5fa', '#2563eb', '#1e40af')


class HistoryCalendar(QWidget):
    date_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected = None
        self.today = None
        self.follow_today = True
        self.cells = {}
        self._signature = None
        self.data = None
        self._range = None
        self._weeks = 0
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        header = QHBoxLayout()
        self.caption = QLabel()
        self.caption.setStyleSheet('font-weight:600;color:#0f172a;')
        header.addWidget(self.caption)
        header.addStretch(1)
        legend = QHBoxLayout()
        legend.setSpacing(4)
        legend.addWidget(QLabel('少'))
        for color in COLORS:
            key = QLabel()
            key.setFixedSize(10, 10)
            key.setStyleSheet(f'background:{color};border-radius:2px;')
            legend.addWidget(key)
        legend.addWidget(QLabel('多'))
        header.addLayout(legend)
        layout.addLayout(header)
        self.grid = QGridLayout()
        self.grid.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        self.grid.setHorizontalSpacing(3)
        self.grid.setVerticalSpacing(3)
        line = QHBoxLayout()
        line.addLayout(self.grid)
        line.addStretch(1)
        layout.addLayout(line)

    def set_data(self, data):
        signature = (data['today'], data['start'], repr(data['days']))
        if signature == self._signature:
            return
        self._signature = signature
        old_selected = self.selected
        self.data = data
        self.today = date.fromisoformat(data['today'])
        start, end = date.fromisoformat(data['start']), date.fromisoformat(data['end'])
        if (self.selected is None or self.follow_today
                or not start <= self.selected <= self.today):
            self.selected = self.today
            self.follow_today = True
        self.caption.setText(f'{start.year}年{start.month}月 — {self.today.year}年{self.today.month}月'
                             if start.year != self.today.year else
                             f'{start.year}年 · {start.month}—{self.today.month}月')
        if self._range != (start, self.today):
            self._range = start, self.today
            self._rebuild_grid(start)
        self._fit_cells()
        self.paint_cells()
        if old_selected != self.selected:
            self.date_selected.emit(self.selected.isoformat())

    def _rebuild_grid(self, start):
        while self.grid.count():
            self.grid.takeAt(0).widget().deleteLater()
        self.cells = {}
        first = start - timedelta(days=start.weekday())
        self._weeks = (self.today-first).days//7 + 1
        for row, text in enumerate('一二三四五六日', 1):
            label = QLabel(text)
            label.setStyleSheet('color:#94a3b8;font-size:11px;')
            self.grid.addWidget(label, row, 0)
        day = start
        while day <= self.today:
            column = (day-first).days//7 + 1
            if day.day == 1:
                label = QLabel(f'{day.month}月')
                label.setStyleSheet('color:#64748b;font-size:12px;')
                self.grid.addWidget(label, 0, column, 1, 3 if self._weeks > 30 else 2)
            cell = QToolButton()
            cell.setFixedSize(16, 16)
            cell.setAccessibleName(day.isoformat())
            cell.clicked.connect(lambda _checked=False, value=day: self.select(value))
            self.grid.addWidget(cell, day.weekday()+1, column)
            self.cells[day] = cell
            day += timedelta(days=1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_cells()

    def _fit_cells(self):
        if not self._weeks:
            return
        gap = 2 if self._weeks > 30 else 3
        side = max(7, min(16, (self.contentsRect().width()-18-gap*self._weeks)//self._weeks))
        self.grid.setHorizontalSpacing(gap)
        for cell in self.cells.values():
            if cell.width() != side:
                cell.setFixedSize(side, side)

    def select(self, day):
        if day is None or day not in self.cells or day > self.today:
            return
        self.selected = day
        self.follow_today = day == self.today
        self.paint_cells()
        self.date_selected.emit(day.isoformat())

    def paint_cells(self):
        for day, cell in self.cells.items():
            info = self.data['days'].get(day.isoformat(), {})
            seconds = info.get('seconds', 0.)
            color = COLORS[heat_level(seconds)]
            border = ('2px solid #0f172a' if day == self.selected else
                      '1px solid #60a5fa' if day == self.today else '1px solid transparent')
            cell.setStyleSheet(f'QToolButton{{background:{color};border:{border};border-radius:3px;}}'
                              'QToolButton:hover{border:1px solid #2563eb;}'
                              'QToolButton:focus{border:2px solid #0f172a;}')
            if info:
                text = f'{day.isoformat()} · 累计{minutes_text(seconds)} · {info.get("tasks", 0)}个任务'
                if info.get('demo_tasks'):
                    text += f'\n另有{info["demo_tasks"]}个模拟任务，未计入累计'
                if info.get('unknown_tasks'):
                    text += '\n身份异常记录未计入累计'
                if info.get('unknown_duration_tasks'):
                    text += '\n存在时长未知的旧记录，累计仅含已知时长'
                if info.get('legacy'):
                    text += '\n部分旧记录缺少完整计时分段，详见任务提示'
            else:
                text = f'{day.isoformat()} · 暂无记录'
            cell.setToolTip(text)
            cell.setAccessibleDescription(text)

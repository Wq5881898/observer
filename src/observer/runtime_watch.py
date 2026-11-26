"""Conservative continuity guard; does not claim to identify OS sleep."""

WORKER_STALL_REASON = '观察循环长时间未响应'


class RuntimeWatch:
    def __init__(self, max_gap=10., clock_tolerance=5.):
        self.max_gap = max_gap
        self.clock_tolerance = clock_tolerance
        self.previous = None
        self.elapsed = 0.
        self.wall_elapsed = 0.

    def check(self, monotonic, wall):
        previous, self.previous = self.previous, (monotonic, wall)
        if previous is None:
            return None
        self.elapsed = monotonic - previous[0]
        self.wall_elapsed = wall - previous[1]
        if (self.elapsed < 0
                or abs(self.wall_elapsed - self.elapsed) > self.clock_tolerance):
            return '检测到系统时间变化或睡眠恢复'
        if self.elapsed > self.max_gap:
            return WORKER_STALL_REASON
        return None

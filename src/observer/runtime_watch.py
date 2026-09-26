"""Conservative continuity guard; does not claim to identify OS sleep."""


class RuntimeWatch:
    def __init__(self, max_gap=10., clock_tolerance=5.):
        self.max_gap = max_gap
        self.clock_tolerance = clock_tolerance
        self.previous = None

    def check(self, monotonic, wall):
        previous, self.previous = self.previous, (monotonic, wall)
        if previous is None:
            return None
        elapsed = monotonic - previous[0]
        wall_elapsed = wall - previous[1]
        if elapsed < 0 or abs(wall_elapsed - elapsed) > self.clock_tolerance:
            return '检测到系统时间变化或睡眠恢复'
        if elapsed > self.max_gap:
            return '检测到运行长时间中断（可能为睡眠或设备阻塞）'
        return None

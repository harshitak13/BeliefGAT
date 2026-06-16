from __future__ import annotations


class CurriculumUpdater:
    def __init__(self, interval: int = 1000):
        self.interval = interval

    def should_update(self, global_step: int) -> bool:
        return global_step > 0 and global_step % self.interval == 0

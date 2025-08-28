
# utils.py
from collections import defaultdict
import time

class Inventory:
    def __init__(self, deplete_seconds=3600):
        self.items = defaultdict(lambda: {"count":0, "last_seen":0})
        self.deplete_seconds = deplete_seconds

    def update_from_detections(self, detections):
        ts = time.time()
        per_class = {}
        for cls, conf, bbox in detections:
            per_class.setdefault(cls, 0)
            per_class[cls] += 1

        for cls, cnt in per_class.items():
            self.items[cls]["count"] = max(self.items[cls]["count"], cnt)
            self.items[cls]["last_seen"] = ts

    def decay(self):
        ts = time.time()
        to_delete = []
        for cls, v in self.items.items():
            if ts - v["last_seen"] > self.deplete_seconds:
                to_delete.append(cls)
        for cls in to_delete:
            del self.items[cls]

    def get_inventory_summary(self):
        return {cls: v["count"] for cls, v in self.items.items()}

    def set_manual(self, cls, count):
        self.items[cls]["count"] = count
        self.items[cls]["last_seen"] = time.time()

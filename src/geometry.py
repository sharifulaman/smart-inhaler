"""Pure-numpy geometry helpers (no heavy dependencies, easy to unit test)."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

NAN = float("nan")
Point = Tuple[float, float]


def wrap_axis_deg(deg: float) -> float:
    """Wrap an undirected axis angle into [-90, 90)."""
    return ((deg + 90.0) % 180.0) - 90.0


def axis_angle_deg(p1: Sequence[float], p2: Sequence[float]) -> float:
    """Angle of the undirected line p1-p2 vs. the image x axis, in [-90, 90)."""
    return wrap_axis_deg(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))


def axis_diff_deg(a: float, b: float) -> float:
    """Signed smallest difference a - b between two undirected axes, in [-90, 90)."""
    return wrap_axis_deg(a - b)


def unwrap_axis(prev: float, new: float) -> float:
    """Shift `new` by multiples of 180 deg so it is continuous with `prev`."""
    while new - prev > 90.0:
        new -= 180.0
    while new - prev < -90.0:
        new += 180.0
    return new


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def point_box_dist(pt: Sequence[float], box: Sequence[float]) -> float:
    """Distance from a point to an axis-aligned box (0 if inside)."""
    x1, y1, x2, y2 = box
    dx = max(x1 - pt[0], 0.0, pt[0] - x2)
    dy = max(y1 - pt[1], 0.0, pt[1] - y2)
    return math.hypot(dx, dy)


def points_box_min_dist(pts: np.ndarray, box: Sequence[float]) -> float:
    pts = np.asarray(pts, dtype=float)
    x1, y1, x2, y2 = box
    dx = np.maximum(np.maximum(x1 - pts[:, 0], 0.0), pts[:, 0] - x2)
    dy = np.maximum(np.maximum(y1 - pts[:, 1], 0.0), pts[:, 1] - y2)
    return float(np.min(np.hypot(dx, dy)))


def obb_long_axis(poly: np.ndarray) -> Tuple[Point, Point, float]:
    """Long axis of an oriented box given as 4 ordered corners.

    Returns (end_a, end_b, length): the midpoints of the two short edges.
    """
    p = np.asarray(poly, dtype=float).reshape(4, 2)
    e01 = float(np.linalg.norm(p[1] - p[0]))
    e12 = float(np.linalg.norm(p[2] - p[1]))
    if e01 >= e12:  # long edges are 0-1 and 2-3; short edges are 1-2 and 3-0
        a, b, length = (p[1] + p[2]) / 2, (p[3] + p[0]) / 2, e01
    else:           # long edges are 1-2 and 3-0; short edges are 0-1 and 2-3
        a, b, length = (p[0] + p[1]) / 2, (p[2] + p[3]) / 2, e12
    return (float(a[0]), float(a[1])), (float(b[0]), float(b[1])), length


def box_long_axis(box: Sequence[float]) -> Tuple[Point, Point, float]:
    """Fallback for axis-aligned boxes: horizontal or vertical (quantised, low quality)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if w >= h:
        return (x1, cy), (x2, cy), float(w)
    return (cx, y1), (cx, y2), float(h)

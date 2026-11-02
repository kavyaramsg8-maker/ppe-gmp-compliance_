"""Per-camera zone geometry, per-zone requirements and shared PPE evidence rules.

Used by the backend (authoritative verdicts), the Mac detector (drawing, handwash
evidence) and tests, so geometry and policy cannot silently drift apart.
No model or web-framework imports: this module must load on Render and on the Mac.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / 'config' / 'camera_zones.yaml'
PPE_ITEMS = ('hairnet', 'mask', 'gloves')
ACTION_CHECKS = ('face_touch', 'handwash_incomplete')
PPE_CLASSES = {'hairnet': ('hairnet', 'present'), 'no_hairnet': ('hairnet', 'missing'),
               'maskon': ('mask', 'present'), 'maskoff': ('mask', 'missing'),
               'glove': ('gloves', 'present'), 'no_glove': ('gloves', 'missing')}
OUTSIDE = 'outside'


def point_in_polygon(point, polygon):
    """Ray casting; points on an edge count as inside. Coordinates in any consistent unit."""
    x, y = point
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # On-edge test: collinear and inside the segment's bounding box.
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) < 1e-12 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


def normalize_point(x, y, width, height, content_box=None):
    """Map a pixel in the received image to normalised camera coordinates.

    content_box=(left, top, right, bottom) in pixels excludes letterbox bars; the
    point is None when it lies on a bar or outside the image.
    """
    left, top, right, bottom = content_box or (0, 0, width, height)
    if right <= left or bottom <= top or not (math.isfinite(x) and math.isfinite(y)):
        return None
    nx, ny = (x - left) / (right - left), (y - top) / (bottom - top)
    if not (-0.01 <= nx <= 1.01 and -0.01 <= ny <= 1.01):
        return None
    return min(1.0, max(0.0, nx)), min(1.0, max(0.0, ny))


def denormalize_polygon(polygon, width, height, content_box=None):
    left, top, right, bottom = content_box or (0, 0, width, height)
    return [(left + x * (right - left), top + y * (bottom - top)) for x, y in polygon]


@dataclass(frozen=True)
class Region:
    zone: str
    polygon: tuple


@dataclass(frozen=True)
class CameraLayout:
    camera_id: str
    default_zone: str
    regions: tuple = ()


@dataclass(frozen=True)
class ZoneRules:
    name: str
    required_ppe: tuple = ()
    action_checks: tuple = ()
    handwash_station: bool = False


@dataclass(frozen=True)
class ZoneConfig:
    cameras: dict
    zones: dict
    handwash: dict = field(default_factory=dict)

    def zone_for(self, camera_id, point):
        """Zone of a normalised foot point; 'outside' when it cannot be placed.

        A camera without regions is one zone, so no foot point is needed there.
        """
        layout = self.cameras.get(camera_id)
        if layout is None:
            return OUTSIDE
        if not layout.regions:
            return layout.default_zone
        if point is None:
            return OUTSIDE
        for region in layout.regions:
            if point_in_polygon(point, region.polygon):
                return region.zone
        return layout.default_zone

    def rules(self, zone):
        return self.zones.get(zone, ZoneRules(zone))

    def station_polygons(self, camera_id):
        layout = self.cameras.get(camera_id)
        if layout is None:
            return []
        return [r.polygon for r in layout.regions if self.rules(r.zone).handwash_station]

    @property
    def handwash_seconds(self):
        return float(self.handwash.get('required_seconds', 20))

    @property
    def max_gap_seconds(self):
        return float(self.handwash.get('max_gap_seconds', 1.0))

    @property
    def forget_after_seconds(self):
        return float(self.handwash.get('forget_after_seconds', 15))

    def to_dict(self):
        return {'cameras': {c: {'default_zone': l.default_zone,
                                'regions': [{'zone': r.zone, 'polygon': [list(p) for p in r.polygon]} for r in l.regions]}
                            for c, l in self.cameras.items()},
                'zones': {z: {'required_ppe': list(r.required_ppe), 'action_checks': list(r.action_checks),
                              'handwash_station': r.handwash_station} for z, r in self.zones.items()},
                'handwash': dict(self.handwash)}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or not isinstance(data.get('cameras'), dict) or not isinstance(data.get('zones'), dict):
            raise ValueError('Zone config needs "cameras" and "zones" mappings')
        zones = {}
        for name, raw in data['zones'].items():
            raw = raw or {}
            required = tuple(raw.get('required_ppe') or ())
            checks = tuple(raw.get('action_checks') or ())
            if unknown := set(required) - set(PPE_ITEMS):
                raise ValueError(f'Zone {name}: unknown PPE {sorted(unknown)}; model supports {PPE_ITEMS}')
            if unknown := set(checks) - set(ACTION_CHECKS):
                raise ValueError(f'Zone {name}: unknown action check {sorted(unknown)}')
            zones[name] = ZoneRules(name, required, checks, bool(raw.get('handwash_station', False)))
        cameras = {}
        for camera_id, raw in data['cameras'].items():
            raw = raw or {}
            default = raw.get('default_zone')
            if default not in zones:
                raise ValueError(f'{camera_id}: default_zone {default!r} is not a configured zone')
            regions = []
            for region in raw.get('regions') or ():
                if region.get('zone') not in zones:
                    raise ValueError(f'{camera_id}: region zone {region.get("zone")!r} is not configured')
                polygon = tuple((float(x), float(y)) for x, y in region.get('polygon') or ())
                if len(polygon) < 3 or not all(0 <= v <= 1 for p in polygon for v in p):
                    raise ValueError(f'{camera_id}/{region["zone"]}: polygon needs >=3 normalised [x, y] points within 0..1')
                regions.append(Region(region['zone'], polygon))
            cameras[camera_id] = CameraLayout(camera_id, default, tuple(regions))
        handwash = dict(data.get('handwash') or {})
        for key in ('required_seconds', 'max_gap_seconds', 'forget_after_seconds'):
            if key in handwash and not 0 < float(handwash[key]) <= 600:
                raise ValueError(f'handwash.{key} must be between 0 and 600 seconds')
        return cls(cameras, zones, handwash)


def load_zone_config(path=None):
    """Load YAML/JSON layout. PPE_HANDWASH_ROI keeps the earlier Render setting working."""
    path = Path(path or os.environ.get('PPE_ZONE_CONFIG') or DEFAULT_PATH)
    text = path.read_text()
    if path.suffix == '.json':
        data = json.loads(text)
    else:
        import yaml
        data = yaml.safe_load(text)
    roi = os.environ.get('PPE_HANDWASH_ROI')
    if roi:
        left, top, right, bottom = (float(v) for v in json.loads(roi))
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError('PPE_HANDWASH_ROI must be [left, top, right, bottom] between 0 and 1')
        for camera in data['cameras'].values():
            for region in camera.get('regions') or ():
                if (data['zones'].get(region['zone']) or {}).get('handwash_station'):
                    region['polygon'] = [[left, top], [right, top], [right, bottom], [left, bottom]]
    polygon = os.environ.get('PPE_HANDWASH_POLYGON')
    if polygon:
        # Calibrated sink polygon (normalised [[x, y], ...]); validated by from_dict.
        for camera in data['cameras'].values():
            for region in camera.get('regions') or ():
                if (data['zones'].get(region['zone']) or {}).get('handwash_station'):
                    region['polygon'] = json.loads(polygon)
    return ZoneConfig.from_dict(data)


def select_ppe(detections, threshold=0.5, conflict_margin=0.10):
    """At most one scored state per PPE item.

    Below-threshold scores are dropped. When the best positive and best negative for
    one item are both above threshold and within conflict_margin, the item is left
    unknown rather than guessed.
    """
    best = {}
    for d in detections:
        name = d.get('class') if isinstance(d, dict) else d
        score = float(d.get('confidence', 0)) if isinstance(d, dict) else float('nan')
        if name not in PPE_CLASSES or not math.isfinite(score) or not threshold <= score <= 1:
            continue
        if name not in best or score > best[name]['confidence']:
            best[name] = {'class': name, 'confidence': score}
    result = []
    for item in PPE_ITEMS:
        choices = sorted((d for name, d in best.items() if PPE_CLASSES[name][0] == item),
                         key=lambda d: d['confidence'], reverse=True)
        if len(choices) > 1 and choices[0]['confidence'] - choices[1]['confidence'] < conflict_margin:
            continue
        if choices:
            result.append(choices[0])
    return result


def ppe_states(detections):
    result = {item: 'unknown' for item in PPE_ITEMS}
    for d in detections:
        name = d.get('class') if isinstance(d, dict) else d
        if name in PPE_CLASSES:
            item, state = PPE_CLASSES[name]
            result[item] = state
    return result


class HandwashTimer:
    """Accumulates only observed, consecutive active-washing evidence.

    Time comes from frame capture timestamps, so processing rate and processing delay
    do not change the measured duration. An interval counts only when both of its
    end samples were active, in the station, for the same track, and closer together
    than max_gap (missing frames, pauses and tracking gaps add nothing).
    """

    def __init__(self, required_seconds=20.0, max_gap=1.0):
        self.required = float(required_seconds)
        self.max_gap = float(max_gap)
        self.elapsed = 0.0
        self.last_time = None
        self.last_active = False
        self.gaps = 0

    @property
    def done(self):
        return self.elapsed >= self.required

    def update(self, timestamp, active):
        if self.last_time is not None and timestamp <= self.last_time:
            return self.elapsed  # stale or duplicate sample: ignore entirely
        if self.last_time is not None and active and self.last_active:
            dt = timestamp - self.last_time
            if dt <= self.max_gap:
                self.elapsed = min(self.required, self.elapsed + dt)
            else:
                self.gaps += 1
        self.last_time, self.last_active = timestamp, bool(active)
        return self.elapsed

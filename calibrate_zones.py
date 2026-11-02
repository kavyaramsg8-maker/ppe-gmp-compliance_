"""Draw configured zones on a still from a camera and click to measure a new polygon.

  python calibrate_zones.py --camera CAM_01 --image live_CAM_01.jpg
  python calibrate_zones.py --camera CAM_01 --image sink.jpg --save-preview zones_CAM_01.png

Click the corners of the sink area in order; the normalised polygon is printed in
camera_zones.yaml format. Keys: u = undo point, c = clear, q = quit. Use a still taken
by the phone in its final fixed position and orientation, without letterbox bars.
"""
import argparse

import cv2
import numpy as np

from cloud_backend.zone_policy import denormalize_polygon, load_zone_config


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--camera', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--config', help='Zone YAML (default: cloud_backend/config/camera_zones.yaml)')
    p.add_argument('--save-preview', help='Write the overlay image and exit (no window)')
    args = p.parse_args()
    zones = load_zone_config(args.config)
    image = cv2.imread(args.image)
    if image is None:
        p.error(f'Cannot read {args.image}')
    h, w = image.shape[:2]
    layout = zones.cameras.get(args.camera)
    if layout is None:
        p.error(f'{args.camera} is not in the zone config')
    clicks = []

    def render():
        out = image.copy()
        for region in layout.regions:
            pts = np.array(denormalize_polygon(region.polygon, w, h), np.int32)
            cv2.polylines(out, [pts], True, (220, 160, 60), 2)
            cv2.putText(out, f'{region.zone} (configured)', tuple(int(v) for v in pts.min(axis=0) + [6, 22]),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (230, 180, 80), 2)
        cv2.putText(out, f'elsewhere: {layout.default_zone}', (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        if clicks:
            pts = np.array([(int(x * w), int(y * h)) for x, y in clicks], np.int32)
            cv2.polylines(out, [pts], len(clicks) > 2, (60, 220, 60), 2)
            for x, y in pts:
                cv2.circle(out, (int(x), int(y)), 4, (60, 220, 60), -1)
        return out

    if args.save_preview:
        cv2.imwrite(args.save_preview, render()); print(f'Wrote {args.save_preview}'); return

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((round(x / w, 3), round(y / h, 3)))
            print(f'point {len(clicks)}: [{clicks[-1][0]:.3f}, {clicks[-1][1]:.3f}]')
            if len(clicks) >= 3:
                print('        polygon: [' + ', '.join(f'[{a:.3f}, {b:.3f}]' for a, b in clicks) + ']')

    cv2.namedWindow('zones'); cv2.setMouseCallback('zones', on_mouse)
    print(__doc__)
    while True:
        cv2.imshow('zones', render())
        key = cv2.waitKey(30) & 255
        if key == ord('q'):
            break
        if key == ord('u') and clicks:
            clicks.pop()
        if key == ord('c'):
            clicks.clear()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

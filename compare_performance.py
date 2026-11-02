"""Build a review table from actual benchmark JSON files."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('optimized',type=Path);a=p.parse_args()
    before=json.loads(a.baseline.read_text());after=json.loads(a.optimized.read_text())
    for field in ['source','camera','measured_frames','warmup_frames']:
        if before[field]!=after[field]:p.error(f'Mismatched {field}; rerun with the same clip/settings')
    def flops(r):
        values=[m.get('gflops') for m in r['models']]
        return sum(values) if all(v is not None for v in values) else None
    metrics=[('YOLO GFLOPs / frame (excludes pose)',flops(before),flops(after)),
             ('Unique YOLO parameters',before['unique_yolo_parameters'],after['unique_yolo_parameters']),
             ('Detector throughput (FPS)',before['detector_throughput_fps'],after['detector_throughput_fps']),
             ('Detector latency mean (ms)',before['detector_latency_ms']['mean'],after['detector_latency_ms']['mean']),
             ('Detector latency p95 (ms)',before['detector_latency_ms']['p95'],after['detector_latency_ms']['p95']),
             ('Peak sampled host RAM (MiB)',before['peak_sampled_process_rss_mib'],after['peak_sampled_process_rss_mib'])]
    print('# Measured detector comparison\n')
    print(f"Devices: {before['device']} → {after['device']}. Camera: {before['camera']}. Measured frames: {before['measured_frames']}.\n")
    if before['device']!=after['device']:print('Device changed: this is a deployment comparison, not an algorithm-only speedup.\n')
    print('| Metric | Baseline | Optimized |\n|---|---:|---:|')
    def fmt(v):return 'Not measured' if v is None else f'{v:,.3f}'
    for name,b,c in metrics:print(f'| {name} | {fmt(b)} | {fmt(c)} |')
    print('\nScope: detector only; network and ESP32 excluded. FLOPs/parameters cover YOLO networks only, not MediaPipe. Input settings are recorded in each JSON. Accuracy must be evaluated separately. See live.summary.json for actual two-camera accepted throughput and capture age.')

if __name__=='__main__':main()

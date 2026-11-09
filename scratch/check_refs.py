import json
import math

with open('gesture_references.json') as f:
    data = json.load(f)

steps = [data['rest_baseline']] + data.get('up_positions', []) + data.get('down_positions', [])
for s in steps:
    lms = s['landmarks_norm']
    t = lms[4]
    idx = lms[8]
    mid = lms[12]
    scale = s['scale']
    d_ti = math.hypot(t['x'] - idx['x'], t['y'] - idx['y'])
    d_tm = math.hypot(t['x'] - mid['x'], t['y'] - mid['y'])
    step_id = s['step_id']
    print(f"{step_id:10s}: scale={scale:.3f} norm_ti={d_ti/scale:.3f} norm_tm={d_tm/scale:.3f}")

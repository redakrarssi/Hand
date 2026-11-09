import sys
sys.path.insert(0, '.')
from app.gesture_engine import GestureEngine

engine = GestureEngine()
state = engine.hand_state

t = 1000.0
dt = 0.033

flick_up_seq = []
for frame in range(10):
    flick_up_seq.append({'palm': (0.5, 0.7), 'wrist': (0.5, 0.85), 'index': (0.5, 0.55), 'middle': (0.5, 0.55)})
for dy in [-0.02, -0.05, -0.08, -0.07]:
    flick_up_seq.append({'palm': (0.5, 0.701), 'wrist': (0.5, 0.85), 'index': (0.5, 0.55 + dy), 'middle': (0.5, 0.55 + dy)})
for dy in [-0.05, -0.03, -0.01, 0.0]:
    flick_up_seq.append({'palm': (0.5, 0.7), 'wrist': (0.5, 0.85), 'index': (0.5, 0.55 + dy), 'middle': (0.5, 0.55 + dy)})
for frame in range(10):
    flick_up_seq.append({'palm': (0.5, 0.7), 'wrist': (0.5, 0.85), 'index': (0.5, 0.55), 'middle': (0.5, 0.55)})

for i, p in enumerate(flick_up_seq):
    curr_t = t + i * dt
    state.palm_positions.append(p['palm'])
    state.wrist_positions.append(p['wrist'])
    state.index_positions.append(p['index'])
    state.middle_positions.append(p['middle'])
    state.timestamps.append(curr_t)
    
    prev_state = state.cycle_state
    res = engine._detect_swipe(state, now=curr_t)
    print(f"Frame {i:2d}: y={p['index'][1]:.3f} state={state.cycle_state} still={state.cycle_still_count} mov={state.cycle_moving_count} up_pk={state.peak_up:.3f} dn_pk={state.peak_down:.3f} res={res.type if res else None}")

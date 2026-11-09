import sys
sys.path.insert(0, '.')
from app.gesture_engine import GestureEngine
from scratch.test_full_engine import make_hand

engine = GestureEngine()
state = engine.hand_state
dt = 0.033
cur_t = 1000.0

print("Running Test A:")
base_wrist = 0.5
for i in range(15):
    y = base_wrist + (i / 15.0) * 0.4
    lm = make_hand(wrist_y=y, mcp_y=y-0.15, tip_y=y-0.22)
    g = engine._detect_swipe(state, lm, now=cur_t)
    print(f"descend F{i:2d}: state={state.cycle_state} still={state.still_count} mov={state.moving_count} base_wrist={state.base_wrist_y} cur_wrist={y:.2f}")
    cur_t += dt

for i in range(15):
    lm = make_hand(wrist_y=0.9, mcp_y=0.75, tip_y=0.68)
    g = engine._detect_swipe(state, lm, now=cur_t)
    print(f"rest    F{i:2d}: state={state.cycle_state} still={state.still_count} mov={state.moving_count} base_wrist={state.base_wrist_y}")
    cur_t += dt

print(f"End of Test A: state={state.cycle_state} still={state.still_count} motion_frames={len(state.motion_frames)}")

import sys
sys.path.insert(0, '.')
from app.gesture_engine import GestureEngine
from scratch.test_full_engine import make_hand

engine = GestureEngine()
state = engine.hand_state
dt = 0.033
cur_t = 1002.0

for i in range(5):
    lm = make_hand(wrist_y=0.9, mcp_y=0.75, tip_y=0.68)
    g = engine._detect_swipe(state, lm, now=cur_t)
    print(f"F{i:2d} REST: state={state.cycle_state} still={state.still_count} mov={state.moving_count} base={state.base_rel_y}")
    cur_t += dt

for dy in [-0.02, -0.05, -0.08, -0.07]:
    lm = make_hand(wrist_y=0.9, mcp_y=0.75, tip_y=0.68 + dy)
    g = engine._detect_swipe(state, lm, now=cur_t)
    print(f"   FLICK: state={state.cycle_state} still={state.still_count} mov={state.moving_count} g={g}")
    cur_t += dt

for dy in [-0.04, -0.02, 0.0]:
    lm = make_hand(wrist_y=0.9, mcp_y=0.75, tip_y=0.68 + dy)
    g = engine._detect_swipe(state, lm, now=cur_t)
    print(f"   RETURN: state={state.cycle_state} still={state.still_count} mov={state.moving_count} g={g}")
    cur_t += dt

for i in range(10):
    lm = make_hand(wrist_y=0.9, mcp_y=0.75, tip_y=0.68)
    g = engine._detect_swipe(state, lm, now=cur_t)
    print(f"F{i:2d} STILL: state={state.cycle_state} still={state.still_count} mov={state.moving_count} g={g}")
    cur_t += dt

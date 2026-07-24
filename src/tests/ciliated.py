"""Tests + visualization for the ciliated motor interface.

Run it directly:

    uv run python src/tests/ciliated.py

It (1) checks the sign conventions of the interface against a synthetic body whose
cilia we place by hand, and (2) renders `src/tests/ciliated.png` with the cilia layout
of a *grown* NeuronNCA body plus a few kinematic trajectories.

Conventions under test (see CiliatedMotorInterface docstring):
  * neuron positions live in [-1, 1]^2; x = left(-)/right(+), y = back(-)/front(+)
  * front cilia -> backward thrust ; back cilia -> forward thrust
  * right cilia -> turn left (+omega, CCW) ; left cilia -> turn right (-omega)
"""

import os
import sys

# make `src` importable when run as a plain script from the repo root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
	sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import struct

from src.devo.core import Body
from src.devo.motor.ciliated import CiliatedMotorInterface
from src.devo.nn.hypernca import NeuronNCA


# ---------------------------------------------------------------------------
# a minimal spatially-embedded neural state so we can place cilia deterministically
# ---------------------------------------------------------------------------
class FakeNeuralState(struct.PyTreeNode):
	x: jax.Array      # [N, 2] neuron positions in [-1, 1]^2
	v: jax.Array      # [N]    activations
	m: jax.Array      # [N]    motor gene expression
	mask: jax.Array   # [N]    alive/present mask


# cardinal border neurons + one interior neuron
#            front       back        left        right       center
_POS = jnp.array([[0., 1.], [0., -1.], [-1., 0.], [1., 0.], [0., 0.]])
FRONT, BACK, LEFT, RIGHT, CENTER = range(5)


def _state(active_idx: list[int], value: float = 1.0) -> FakeNeuralState:
	n = _POS.shape[0]
	v = jnp.zeros(n).at[jnp.array(active_idx)].set(value) if active_idx else jnp.zeros(n)
	return FakeNeuralState(x=_POS, v=v, m=jnp.ones(n), mask=jnp.ones(n))


def _drive(interface: CiliatedMotorInterface, active_idx: list[int], value: float = 1.0):
	"""Return (velocity, omega) the interface produces for the given active cilia."""
	layout = interface.init(_state([]), key=jr.key(0))  # layout is activation-independent
	body = Body(pos=jnp.zeros(2, dtype=jnp.float16), heading=jnp.float16(0.0), size=jnp.float16(2.0))
	# actuate fuses decode+move; the decoded action is exposed via info for exactly this kind of check
	_, energy_loss, _, info = interface.actuate(_state(active_idx, value), layout, body, jr.key(0))
	velocity, omega = float(info["action"][0]), float(info["action"][1])
	return velocity, omega, float(energy_loss)


# ---------------------------------------------------------------------------
# sign-convention checks
# ---------------------------------------------------------------------------
def run_checks() -> bool:
	itf = CiliatedMotorInterface(
		border_size=0.2, thrust_gain=1.0, max_beat=5.0,
		max_velocity=10.0, max_angular_speed=10.0, motor_energy_cost=0.1,
		dt=1.0,
	)

	cases = []

	# --- classification: which neurons are cilia at all ---
	layout = itf.init(_state([]), key=jr.key(0))
	cases.append(("front is a velocity cilium", bool(layout.is_velocity_cilium[FRONT])))
	cases.append(("back is a velocity cilium", bool(layout.is_velocity_cilium[BACK])))
	cases.append(("left is an angular cilium", bool(layout.is_angular_cilium[LEFT])))
	cases.append(("right is an angular cilium", bool(layout.is_angular_cilium[RIGHT])))
	cases.append(("center is NOT a cilium",
	              not bool(layout.is_velocity_cilium[CENTER] or layout.is_angular_cilium[CENTER])))

	# --- velocity sign ---
	v, w, _ = _drive(itf, [BACK]);  cases.append(("back cilium -> forward (v>0)", v > 0 and abs(w) < 1e-4))
	v, w, _ = _drive(itf, [FRONT]); cases.append(("front cilium -> backward (v<0)", v < 0 and abs(w) < 1e-4))

	# --- angular sign ---
	v, w, _ = _drive(itf, [RIGHT]); cases.append(("right cilium -> turn left (omega>0)", w > 0 and abs(v) < 1e-4))
	v, w, _ = _drive(itf, [LEFT]);  cases.append(("left cilium -> turn right (omega<0)", w < 0 and abs(v) < 1e-4))

	# --- beating rate is non-negative (a negatively-activated cilium does not thrust) ---
	v, w, e = _drive(itf, [BACK], value=-1.0)
	cases.append(("negative activation -> no thrust", abs(v) < 1e-4 and abs(w) < 1e-4 and e == 0.0))

	# --- max_beat clipping caps thrust ---
	v, _, _ = _drive(itf, [BACK], value=100.0)
	cases.append(("thrust clipped to max_beat", abs(v - itf.max_beat) < 1e-3))

	# --- energy cost scales with active beating ---
	_, _, e = _drive(itf, [BACK], value=1.0)
	cases.append(("energy cost > 0 when beating", e > 0.0))

	# --- move() kinematics: forward drive translates along heading ---
	body = Body(pos=jnp.zeros(2, dtype=jnp.float16), heading=jnp.float16(0.0), size=jnp.float16(2.0))
	action = jnp.array([1.0, 0.0], dtype=jnp.float16)  # velocity=1, omega=0, heading 0 => +x
	moved = itf.move(action, body)
	cases.append(("move: +velocity at heading 0 -> +x", float(moved.pos[0]) > 0 and abs(float(moved.pos[1])) < 1e-2))

	# --- move() rotation: +omega increases heading (CCW / left turn) ---
	action = jnp.array([0.0, 0.5], dtype=jnp.float16)
	moved = itf.move(action, body)
	cases.append(("move: +omega -> heading increases", float(moved.heading) > 0))

	print("\n=== ciliated interface checks ===")
	ok = True
	for name, passed in cases:
		print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
		ok = ok and passed
	print(f"=== {'ALL PASSED' if ok else 'FAILURES PRESENT'} ===\n")
	return ok


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------
def _rollout(interface, neural_state, layout, steps: int, heading0: float = 0.0):
	body = Body(pos=jnp.zeros(2, dtype=jnp.float16), heading=jnp.float16(heading0), size=jnp.float16(2.0))
	traj = [(float(body.pos[0]), float(body.pos[1]), float(body.heading))]
	for _ in range(steps):
		body, _, _, _ = interface.actuate(neural_state, layout, body, jr.key(0))
		traj.append((float(body.pos[0]), float(body.pos[1]), float(body.heading)))
	return traj


def make_figure(path: str) -> None:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	import numpy as np

	itf = CiliatedMotorInterface(
		border_size=0.2, thrust_gain=1.0, max_beat=1.0,
		max_velocity=10.0, max_angular_speed=jnp.pi / 4, motor_energy_cost=0.1,
		dt=1.0,
	)

	fig = plt.figure(figsize=(16, 5))
	gs = fig.add_gridspec(1, 4)

	# --- Panel A: cilia layout of a grown NeuronNCA body ---
	nca = NeuronNCA(size=16, synapse_channels=8, extra_channels=16, perception_channels=3,
	                update_layers=2, dev_steps=50, nb_wiring_rules=4, key=jr.key(0))
	grown = nca.init(jr.key(1))
	layout = itf.init(grown, key=jr.key(2))

	x = np.asarray(grown.x)
	vel_c = np.asarray(layout.is_velocity_cilium)
	ang_c = np.asarray(layout.is_angular_cilium)
	other = ~(vel_c | ang_c)

	ax = fig.add_subplot(gs[0, 0])
	ax.scatter(x[other, 0], x[other, 1], s=14, c="lightgray", label="non-cilia")
	ax.scatter(x[vel_c, 0], x[vel_c, 1], s=34, c="tab:blue", label="velocity cilia (front/back)")
	ax.scatter(x[ang_c, 0], x[ang_c, 1], s=34, c="tab:red", label="angular cilia (left/right)")
	thr = 1.0 - itf.border_size
	ax.add_patch(plt.Rectangle((-thr, -thr), 2 * thr, 2 * thr, fill=False, ls="--", ec="k", alpha=0.4))
	ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.15), title="grown NeuronNCA: cilia layout",
	       xlabel="x  (left - / right +)", ylabel="y  (back - / front +)")
	ax.set_aspect("equal")
	ax.legend(loc="upper center", fontsize=7, framealpha=0.9)

	# --- Panels B-D: one kinematic trajectory per canonical drive (synthetic body) ---
	drives = [
		("back only  ->  forward", [BACK], "tab:green", 16),
		("back + right  ->  curve left", [BACK, RIGHT], "tab:orange", 24),
		("front + left  ->  curve back-right", [FRONT, LEFT], "tab:purple", 24),
	]
	for col, (label, active, color, steps) in enumerate(drives, start=1):
		ax = fig.add_subplot(gs[0, col])
		st = _state(active, value=1.0)
		lay = itf.init(st, key=jr.key(0))
		traj = _rollout(itf, st, lay, steps=steps, heading0=0.0)
		tx = np.array([p[0] for p in traj])
		ty = np.array([p[1] for p in traj])
		ax.plot(tx, ty, "-", color=color)
		for p in traj[::4]:
			ax.arrow(p[0], p[1], 0.5 * np.cos(p[2]), 0.5 * np.sin(p[2]),
			         head_width=0.18, color=color, alpha=0.55, length_includes_head=True)
		ax.plot(tx[0], ty[0], "o", color="k", ms=6, label="start")
		ax.set(title=label, xlabel="world x", ylabel="world y")
		ax.set_aspect("equal")
		# pad limits a touch so arrows are not clipped
		span = max(tx.max() - tx.min(), ty.max() - ty.min(), 1.0) * 0.15
		ax.set_xlim(tx.min() - span - 0.6, tx.max() + span + 0.6)
		ax.set_ylim(ty.min() - span - 0.6, ty.max() + span + 0.6)
		ax.legend(loc="best", fontsize=7)

	fig.suptitle("ciliated motor interface", fontsize=13)
	fig.tight_layout()
	fig.savefig(path, dpi=110)
	print(f"wrote figure -> {path}")


if __name__ == "__main__":
	ok = run_checks()
	out = os.path.join(os.path.dirname(__file__), "ciliated.png")
	make_figure(out)
	sys.exit(0 if ok else 1)

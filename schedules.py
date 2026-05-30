from __future__ import annotations


def constant_schedule(step_idx: int, num_steps: int, scale: float) -> float:
    return float(scale)


def linear_ramp_schedule(step_idx: int, num_steps: int, scale: float) -> float:
    if num_steps <= 1:
        return float(scale)
    ratio = step_idx / float(num_steps - 1)
    return float(scale) * ratio


def cosine_decay_schedule(step_idx: int, num_steps: int, scale: float) -> float:
    if num_steps <= 1:
        return float(scale)
    import math

    ratio = step_idx / float(num_steps - 1)
    return float(scale) * 0.5 * (1.0 + math.cos(math.pi * ratio))


def get_schedule_value(name: str, step_idx: int, num_steps: int, scale: float) -> float:
    if name == "constant":
        return constant_schedule(step_idx, num_steps, scale)
    if name == "linear_ramp":
        return linear_ramp_schedule(step_idx, num_steps, scale)
    if name == "cosine_decay":
        return cosine_decay_schedule(step_idx, num_steps, scale)
    raise ValueError(f"Unsupported schedule: {name}")

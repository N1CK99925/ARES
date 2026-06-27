class EpsilonScheduler:
    """Linear epsilon decay from a start value to an end value over decay_steps."""

    def __init__(self, start: float, end: float, decay_steps: int) -> None:
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def get_epsilon(self, step: int) -> float:
        # Fraction of decay completed, clamped to [0, 1]
        progress = min(step / self.decay_steps, 1.0)
        # Linear interpolation: at step=0 → start, at step=decay_steps → end
        epsilon = self.start - (self.start - self.end) * progress
        return epsilon

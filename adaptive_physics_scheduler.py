"""
adaptive_physics_scheduler.py

Implements the three adaptive physics-constrained loss scheduling strategies
described in RESEARCH_REPORT.md (Section 5):

  Strategy 1 — LinearWarmupScheduler    : linearly ramps λ_phy from λ_init → λ_max
  Strategy 2 — GradientNormBalancer     : keeps ||∇L_physics|| / ||∇L_MSE|| ≈ 1
  Strategy 3 — ValidationPlateauAnnealer: increments λ_phy when val-loss plateaus

All schedulers expose a common interface:
  scheduler.step(...)  → updates λ_phy and returns the new value
  scheduler.lambda_phy → current weight as a float
  scheduler.history    → list of (epoch, λ_phy, rho) records for plotting
"""

import math
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BasePhysicsScheduler:
    """Shared interface for all adaptive schedulers."""

    def __init__(self, lambda_init: float = 0.01, lambda_min: float = 1e-3,
                 lambda_max: float = 10.0):
        self.lambda_phy = lambda_init
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.history: list[dict] = []   # records per epoch

    def _clip(self, val: float) -> float:
        return max(self.lambda_min, min(self.lambda_max, val))

    def step(self, **kwargs) -> float:
        raise NotImplementedError

    def log_epoch(self, epoch: int, rho: float | None = None):
        self.history.append({"epoch": epoch, "lambda_phy": self.lambda_phy,
                              "rho": rho})
        logger.info(
            f"[PhysicsScheduler] epoch={epoch}  λ_phy={self.lambda_phy:.5f}"
            + (f"  ρ={rho:.3f}" if rho is not None else "")
        )


# ---------------------------------------------------------------------------
# Strategy 1: Linear Warm-Up
# ---------------------------------------------------------------------------

class LinearWarmupScheduler(BasePhysicsScheduler):
    """
    Linearly ramps the physics loss weight from λ_init to λ_max over
    T_warmup epochs, then holds it at λ_max.

    Schedule:
        λ(e) = λ_max * min(1.0, e / T_warmup)

    Args:
        lambda_init  : starting weight (default 0.01)
        lambda_max   : target/ceiling weight (default 1.0)
        T_warmup     : number of epochs to reach λ_max (default 20)
    """

    def __init__(self, lambda_init: float = 0.01, lambda_max: float = 1.0,
                 T_warmup: int = 20):
        super().__init__(lambda_init=lambda_init, lambda_max=lambda_max)
        self.T_warmup = T_warmup
        self._epoch = 0

    def step(self, **kwargs) -> float:
        """Call once per epoch (no gradient info needed)."""
        self._epoch += 1
        new_val = self.lambda_max * min(1.0, self._epoch / self.T_warmup)
        self.lambda_phy = self._clip(new_val)
        self.log_epoch(self._epoch)
        return self.lambda_phy


# ---------------------------------------------------------------------------
# Strategy 2: Gradient Norm Balancing (GNB)
# ---------------------------------------------------------------------------

class GradientNormBalancer(BasePhysicsScheduler):
    """
    Dynamically adjusts λ_phy so that the physics gradient magnitude matches
    the MSE gradient magnitude, keeping ρ = ||∇L_phy|| / ||∇L_mse|| ≈ 1.

    Update rule (applied once per epoch):
        λ_phy ← λ_phy  *  (g_mse / (g_phy + ε))

    where g_mse and g_phy are the epoch-averaged gradient norms for each loss
    component, computed via separate backward passes.

    Args:
        lambda_init  : starting weight (default 0.1)
        lambda_min   : floor (default 1e-3)
        lambda_max   : ceiling (default 10.0)
        eps          : numerical stability (default 1e-8)
        smooth       : EMA smoothing factor for gradient norms (default 0.9)
    """

    def __init__(self, lambda_init: float = 0.1, lambda_min: float = 1e-3,
                 lambda_max: float = 10.0, eps: float = 1e-8,
                 smooth: float = 0.9):
        super().__init__(lambda_init=lambda_init, lambda_min=lambda_min,
                         lambda_max=lambda_max)
        self.eps = eps
        self.smooth = smooth
        self._ema_g_mse: float | None = None
        self._ema_g_phy: float | None = None
        self._epoch = 0

    def _update_ema(self, current_g_mse: float, current_g_phy: float):
        if self._ema_g_mse is None:
            self._ema_g_mse = current_g_mse
            self._ema_g_phy = current_g_phy
        else:
            self._ema_g_mse = (self.smooth * self._ema_g_mse
                               + (1 - self.smooth) * current_g_mse)
            self._ema_g_phy = (self.smooth * self._ema_g_phy
                               + (1 - self.smooth) * current_g_phy)

    def step(self, g_mse: float, g_phy: float, **kwargs) -> float:
        """
        Call once per epoch with the epoch-averaged gradient norms.

        Args:
            g_mse : mean ||∇_θ L_MSE|| over all batches this epoch
            g_phy : mean ||∇_θ L_physics|| over all batches this epoch
        """
        self._epoch += 1
        self._update_ema(g_mse, g_phy)

        rho = self._ema_g_phy / (self._ema_g_mse + self.eps)
        # Scale λ_phy so that ρ → 1
        new_val = self.lambda_phy * (self._ema_g_mse
                                     / (self._ema_g_phy + self.eps))
        self.lambda_phy = self._clip(new_val)
        self.log_epoch(self._epoch, rho=rho)
        return self.lambda_phy


# ---------------------------------------------------------------------------
# Strategy 3: Validation Plateau Annealing
# ---------------------------------------------------------------------------

class ValidationPlateauAnnealer(BasePhysicsScheduler):
    """
    Increases λ_phy only when the validation loss has plateaued (stopped
    improving beyond threshold τ), implementing a curriculum-style schedule.

    Update rule:
        if |ΔL_val| < τ  for `patience` consecutive epochs:
            λ_phy ← min(λ_phy * (1 + α), λ_max)
            reset plateau counter

    Args:
        lambda_init   : starting weight (default 0.01)
        lambda_max    : ceiling (default 1.0)
        patience      : epochs of no improvement before triggering (default 5)
        tau           : improvement threshold (default 1e-4)
        alpha         : fractional increment per trigger (default 0.2)
    """

    def __init__(self, lambda_init: float = 0.01, lambda_max: float = 1.0,
                 patience: int = 5, tau: float = 1e-4, alpha: float = 0.2):
        super().__init__(lambda_init=lambda_init, lambda_max=lambda_max)
        self.patience = patience
        self.tau = tau
        self.alpha = alpha
        self._best_val_loss = math.inf
        self._plateau_count = 0
        self._epoch = 0

    def step(self, val_loss: float, **kwargs) -> float:
        """
        Call once per epoch with the current validation loss.

        Args:
            val_loss : validation set total loss for this epoch
        """
        self._epoch += 1
        delta = self._best_val_loss - val_loss

        if delta > self.tau:
            self._best_val_loss = val_loss
            self._plateau_count = 0
        else:
            self._plateau_count += 1

        if self._plateau_count >= self.patience:
            new_val = self.lambda_phy * (1 + self.alpha)
            self.lambda_phy = self._clip(new_val)
            self._plateau_count = 0
            logger.info(
                f"[PlateauAnnealer] Plateau detected. "
                f"λ_phy increased to {self.lambda_phy:.5f}"
            )

        self.log_epoch(self._epoch)
        return self.lambda_phy


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_scheduler(strategy: str, **kwargs) -> BasePhysicsScheduler:
    """
    Factory function. Instantiate by name from config.

    Args:
        strategy : one of "linear_warmup", "gnb", "plateau", "fixed"
        **kwargs : passed to the chosen scheduler constructor

    Returns:
        A BasePhysicsScheduler instance.

    Example:
        sched = build_scheduler("gnb", lambda_init=0.1, lambda_max=5.0)
    """
    registry = {
        "linear_warmup": LinearWarmupScheduler,
        "gnb":           GradientNormBalancer,
        "plateau":       ValidationPlateauAnnealer,
    }
    if strategy == "fixed":
        # Return a trivial scheduler that never changes λ
        class FixedScheduler(BasePhysicsScheduler):
            def step(self, **kwargs):
                self._epoch = getattr(self, "_epoch", 0) + 1
                self.log_epoch(self._epoch)
                return self.lambda_phy
        lam = kwargs.pop("lambda_phy", 1.0)
        return FixedScheduler(lambda_init=lam, lambda_max=lam)

    if strategy not in registry:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from {list(registry.keys()) + ['fixed']}"
        )
    return registry[strategy](**kwargs)

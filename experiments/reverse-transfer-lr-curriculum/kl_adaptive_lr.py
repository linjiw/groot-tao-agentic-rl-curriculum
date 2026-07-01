"""Framework-agnostic KL-adaptive learning-rate controller.

Ported from SONIC (gear_sonic) PPO trainer's ``_adjust_learning_rate_based_on_kl``.

[verified] SOURCE:
  external/GR00T-WholeBodyControl/gear_sonic/trl/trainer/ppo_trainer.py
  lines 2142-2166, method ``_adjust_learning_rate_based_on_kl(self, kl_mean, optimizer)``.

  Exact SONIC logic (copied from source this task):

      if self.desired_kl is None:
          return
      if kl_mean > self.desired_kl * 2.0:
          new_lr = max(self.adaptive_lr_min, self.args.learning_rate / 1.5)
      elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
          new_lr = min(self.adaptive_lr_max, self.args.learning_rate * 1.5)
      else:
          new_lr = self.args.learning_rate
      self.args.learning_rate = new_lr
      for param_group in optimizer.param_groups:
          param_group["lr"] = self.args.learning_rate

This module reproduces that logic EXACTLY, extracted into a self-contained,
domain-agnostic class with a pure-python core (no torch required for the LR
math). If a torch (or torch-like) optimizer is supplied to ``update`` its
``param_groups`` are mutated in place, mirroring the SONIC trainer.

Two SONIC-specific fields are generalized here:
  * ``self.desired_kl`` / ``self.args.learning_rate`` -> constructor args.
  * SONIC hard-codes ``1.5`` as both the shrink divisor and the grow
    multiplier; exposed here as ``factor`` (default 1.5) so the rule stays
    identical to SONIC out of the box but is tunable for TAO.
"""

from __future__ import annotations

from typing import Any, Optional


class KLAdaptiveLR:
    """KL-adaptive learning-rate controller (SONIC port).

    The controller keeps a target KL "band" ``[desired_kl/2, desired_kl*2]``.
    When the observed mean KL between successive policies leaves that band the
    learning rate is multiplicatively adjusted and clamped to ``[lr_min, lr_max]``:

      * ``kl_mean > desired_kl * 2.0``           -> shrink: ``lr / factor`` (clamped at ``lr_min``)
      * ``kl_mean < desired_kl / 2.0`` and ``> 0`` -> grow:   ``lr * factor`` (clamped at ``lr_max``)
      * otherwise                                -> hold:   ``lr`` unchanged

    The ``kl_mean > 0.0`` guard on the grow branch is faithful to SONIC: a
    non-positive KL (e.g. a numerical artifact or an unset value) will NOT grow
    the learning rate.

    If ``desired_kl`` is ``None`` the controller is inert and ``update`` returns
    the current lr unchanged (mirrors SONIC's ``if self.desired_kl is None: return``).
    """

    def __init__(
        self,
        desired_kl: Optional[float],
        lr_min: float,
        lr_max: float,
        lr: float,
        factor: float = 1.5,
    ) -> None:
        """Initialize the controller.

        Args:
            desired_kl: Target KL divergence. ``None`` disables adaptation.
            lr_min: Lower clamp for the learning rate (``adaptive_lr_min`` in SONIC).
            lr_max: Upper clamp for the learning rate (``adaptive_lr_max`` in SONIC).
            lr: Current learning rate (``args.learning_rate`` in SONIC).
            factor: Shrink divisor / grow multiplier. SONIC hard-codes 1.5.
        """
        self.desired_kl = desired_kl
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr = lr
        self.factor = factor

    def update(self, kl_mean: float, optimizer: Optional[Any] = None) -> float:
        """Adjust the learning rate based on the observed mean KL.

        Reproduces SONIC ppo_trainer.py:2154-2166 exactly.

        Args:
            kl_mean: Mean KL divergence between successive policies.
            optimizer: Optional torch(-like) optimizer. If given, every entry in
                ``optimizer.param_groups`` has its ``"lr"`` set to the new lr.

        Returns:
            The (possibly unchanged) new learning rate.
        """
        # [verified] SONIC: if self.desired_kl is None: return
        if self.desired_kl is None:
            return self.lr

        # [verified] SONIC branch logic, ppo_trainer.py:2157-2162
        if kl_mean > self.desired_kl * 2.0:
            new_lr = max(self.lr_min, self.lr / self.factor)
        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            new_lr = min(self.lr_max, self.lr * self.factor)
        else:
            new_lr = self.lr

        # [verified] SONIC: self.args.learning_rate = new_lr
        self.lr = new_lr

        # [verified] SONIC: for param_group in optimizer.param_groups: param_group["lr"] = ...
        if optimizer is not None:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr

        return self.lr

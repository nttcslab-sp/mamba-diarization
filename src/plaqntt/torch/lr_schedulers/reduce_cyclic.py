# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024

import math
import warnings
from typing import Literal
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau


class ReduceCyclicOnPlateau(ReduceLROnPlateau, LRScheduler):
    r"""Combines CyclicLR and ReduceLROnPlateau, compatible with lightning.
    The code is horrendous because we need to inherit from ReduceLROnPlateau as lightning
    only passes metric to the scheduler when `isinstance(scheduler, ReduceLROnPlateau)`.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then 'mode' is ignored.
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
        plateau_patience (int): Number of CYCLES with no improvement after which
            learning rate will be reduced. < 0 values to disable.
            The LR will decrease after `plateau_patience` cycles with no improvement.
            Default: -1
        plateau_factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor.
            Default: 0.5
        plateau_mode (str): One of {min, max}. Defines the whether the metric is
            supposed to be minimized or maximized.
            Default: 'min'
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.
    """

    def __init__(
        self,
        optimizer,
        base_lr,
        max_lr,
        step_size_up=2000,
        step_size_down=None,
        mode="triangular",
        gamma=1.0,
        scale_fn=None,
        scale_mode="cycle",
        cycle_momentum=True,
        base_momentum=0.8,
        max_momentum=0.9,
        last_epoch=-1,
        plateau_patience: int = -1,
        plateau_factor: float = 0.5,
        plateau_mode: Literal["min", "max"] = "min",
    ):

        super(ReduceCyclicOnPlateau, self).__init__(
            optimizer, mode=plateau_mode, factor=plateau_factor, patience=plateau_patience
        )

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        base_lrs = self._format_param("base_lr", optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group["lr"] = lr

        self.max_lrs = self._format_param("max_lr", optimizer, max_lr)

        step_size_up = float(step_size_up)
        step_size_down = float(step_size_down) if step_size_down is not None else step_size_up
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        if mode not in ["triangular", "triangular2", "exp_range"] and scale_fn is None:
            raise ValueError("mode is invalid and scale_fn is None")

        self.mode = mode
        self.gamma = gamma

        self._scale_fn_ref = None
        self._scale_fn_custom = scale_fn
        self.scale_mode = scale_mode
        self._init_scale_fn()

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if "momentum" not in optimizer.defaults and "betas" not in optimizer.defaults:
                raise ValueError("optimizer must support momentum or beta1 with `cycle_momentum` option enabled")

            self.use_beta1 = "betas" in self.optimizer.defaults
            self.base_momentums = self._format_param("base_momentum", optimizer, base_momentum)
            self.max_momentums = self._format_param("max_momentum", optimizer, max_momentum)
            if last_epoch == -1:
                for m_momentum, b_momentum, group in zip(
                    self.max_momentums, self.base_momentums, optimizer.param_groups
                ):
                    if self.use_beta1:
                        group["betas"] = (m_momentum, *group["betas"][1:])
                    else:
                        group["momentum"] = m_momentum
                    group["max_momentum"] = m_momentum
                    group["base_momentum"] = b_momentum

        # new
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        self.plateau_mode = plateau_mode
        self.last_best_metric = (None, None)
        """Tuple (cycle, best_metric)"""

        super(ReduceLROnPlateau, self).__init__(optimizer, last_epoch)
        # super(ReduceLROnPlateau, self).__init__(optimizer, mode=plateau_mode, factor=plateau_factor, patience=plateau_patience)
        self.base_lrs = base_lrs

    def _init_scale_fn(self):
        if self._scale_fn_custom is not None:
            return
        if self.mode == "triangular":
            self._scale_fn_ref = self._triangular_scale_fn
            self.scale_mode = "cycle"
        elif self.mode == "triangular2":
            self._scale_fn_ref = self._triangular2_scale_fn
            self.scale_mode = "cycle"
        elif self.mode == "exp_range":
            self._scale_fn_ref = partial(self._exp_range_scale_fn, self.gamma)
            self.scale_mode = "iterations"

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} values for {name}, got {len(param)}")
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def scale_fn(self, x):
        if self._scale_fn_custom is not None:
            return self._scale_fn_custom(x)
        else:
            return self._scale_fn_ref(x)  # static method

    @staticmethod
    def _triangular_scale_fn(x):
        return 1.0

    @staticmethod
    def _triangular2_scale_fn(x):
        return 1 / (2.0 ** (x - 1))

    @staticmethod
    def _exp_range_scale_fn(gamma, x):
        return gamma**x

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """

        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1.0 + self.last_epoch / self.total_size - cycle

        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == "cycle":
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == "cycle":
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(self.last_epoch)
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                if self.use_beta1:
                    param_group["betas"] = (momentum, *param_group["betas"][1:])
                else:
                    param_group["momentum"] = momentum

        return lrs

    def state_dict(self):
        state = super().state_dict()
        # We are dropping the `_scale_fn_ref` attribute because it is a
        # `weakref.WeakMethod` and can't be pickled.
        state.pop("_scale_fn_ref")
        fn = state.pop("_scale_fn_custom")
        state["_scale_fn_custom"] = None
        if fn is not None and not isinstance(fn, types.FunctionType):
            # The _scale_fn_custom will only be saved if it is a callable object
            # and not if it is a function or lambda.
            state["_scale_fn_custom"] = fn.__dict__.copy()

        return state

    def load_state_dict(self, state_dict):
        fn = state_dict.pop("_scale_fn_custom")
        super().load_state_dict(state_dict)
        if fn is not None:
            self._scale_fn_custom.__dict__.update(fn)
        self._init_scale_fn()

    def get_best(self, metric1, metric2):
        if self.plateau_mode == "min":
            return min(metric1, metric2)
        elif self.plateau_mode == "max":
            return max(metric1, metric2)
        else:
            raise ValueError(f"Unknown plateau_mode {self.plateau_mode}")

    def step(self, metrics):
        # ---- new logic
        current = float(metrics)

        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1.0 + self.last_epoch / self.total_size - cycle
        if self.plateau_patience >= 0 and x == 0:
            print(f"Checking plateau at {cycle} with metric {current} vs {self.last_best_metric[1]}")
            if self.last_best_metric[0] is None or self.get_best(current, self.last_best_metric[1]) == current:
                self.last_best_metric = (cycle, current)
                print(f"New best at {cycle} with metric {current}")
            elif cycle - self.last_best_metric[0] > self.plateau_patience:
                print(f"Plateau detected at {cycle} with metric {current}, reducing LR by {self.plateau_factor}")
                for lr_idx, lr in enumerate(self.max_lrs):
                    self.max_lrs[lr_idx] *= self.plateau_factor

        # -------- regular step logic
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the learning rate schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )
        self._step_count += 1

        with _enable_get_lr_call(self):
            self.last_epoch += 1
            values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group["lr"] = lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _initial_step(self):
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step(math.inf if self.plateau_mode == "min" else -math.inf)


class _enable_get_lr_call:
    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False

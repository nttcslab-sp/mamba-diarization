# Copyright (c) 2024 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Alexis Plaquet, 2024

import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.optimizer import Optimizer


class DecreasingCAWR(CosineAnnealingWarmRestarts):
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        restart_decay: float = 1.0,
    ) -> None:
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch, verbose="deprecated")
        self.restart_decay = restart_decay

    def get_lr(self) -> list[float]:
        # The current cycle. Starts at 0 and increase by 1 at each restart
        if self.T_mult == 1:
            current_cycle = self.last_epoch // self.T_0
        else:
            current_cycle = math.log(self.T_i / self.T_0, self.T_mult)
        decay_factor: float = self.restart_decay**current_cycle

        lrs = [
            self.eta_min + (base_lr - self.eta_min) * decay_factor * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

        return lrs

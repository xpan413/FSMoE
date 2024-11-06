from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.transformer.moe.moe_layer import MoELayer


__all__ = ["CallbackBase"]


class CallbackBase:
    def before_moe_start_hook(self, moe: MoELayer, batch: dict) -> None:
        pass

    def before_dispatch_hook(self, moe: MoELayer, batch: dict) -> None:
        pass

    def after_dispatch_hook(self, moe: MoELayer, batch: dict) -> None:
        pass

    def before_combine_hook(self, moe: MoELayer, batch: dict) -> None:
        pass

    def after_combine_hook(self, moe: MoELayer, batch: dict) -> None:
        pass

    def before_moe_end_hook(self, moe: MoELayer, batch: dict) -> None:
        pass

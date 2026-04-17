from __future__ import annotations

from typing import List, Union

from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = Union[LossDepthCfgWrapper, LossLpipsCfgWrapper, LossMseCfgWrapper]


def get_losses(cfgs: List[LossCfgWrapper]) -> List[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]

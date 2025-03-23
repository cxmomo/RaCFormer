# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from .utils import is_parallel

__all__ = ['SequentialControlHook', 'HisInfoControlHook']


@HOOKS.register_module()
class SequentialControlHook(Hook):
    """ """

    def __init__(self, start_epoch=1):
        super().__init__()
        self.start_epoch=start_epoch

    def set_temporal_flag(self, runner):
        if is_parallel(runner.model.module):
            runner.model.module.module.img_lss_view_transformer.loss_depth_weight = 1.0
        else:
            runner.model.module.img_lss_view_transformer.loss_depth_weight = 1.0
            

    def before_run(self, runner):
        pass

    def before_train_epoch(self, runner):
        if runner.epoch >= self.start_epoch:
            self.set_temporal_flag(runner)

@HOOKS.register_module()
class HisInfoControlHook(Hook):
    """ """

    def __init__(self, hisinfo_start_epoch=-1):
        super().__init__()
        self.his_info_start_epoch=hisinfo_start_epoch

    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.use_his_info=flag
        else:
            runner.model.module.use_his_info = flag

    def before_run(self, runner):
        self.set_temporal_flag(runner, False)

    def before_train_epoch(self, runner):
        if runner.epoch > self.his_info_start_epoch:
            print(f"Open His Info")
            self.set_temporal_flag(runner, True)
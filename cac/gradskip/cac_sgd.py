# cac_sgd.py COPYRIGHT Fujitsu Limited 2021
import os
import ast
import torch
from torch.optim.optimizer import Optimizer, required


class SGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".
                             format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero "
                             "dampening")
        super(SGD, self).__init__(params, defaults)

        self.org_param_groups = list(params)
        self.optimizer_itr = 0

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def is_gs_available(self):
        return (self.is_manual_stop() or self.cac_var_start_thr != -1)

    def is_manual_stop(self):
        return (len(self.cac_stop_layer_num) > 0 and
                len(self.cac_stop_layer_itr) > 0)

    def manual_stop(self):
        skip_decision = False
        try:
            stop_itr = self.cac_stop_layer_itr[self.skipped_count]
            if self.optimizer_itr == stop_itr:
                skip_decision = True
        except IndexError:
            pass
        return skip_decision

    def thresh_stop(self, group, skipped_count=None):
        skip_decision = False
        try:
            if skipped_count is None:
                skipped_count = self.skipped_count
            if (self.cac_var_start_thr != -1
                    and self.optimizer_itr >= self.cac_var_start_itr
                    and self.optimizer_itr % self.cac_var_samples == 0):
                layer_id = self.cac_stop_layer_num[skipped_count]
                p_w_var = torch.var(group['params'][layer_id])
                if p_w_var >= self.w_var_peak:
                    self.w_var_peak = p_w_var
                    self.w_var_mt_count += 1
                else:
                    w_var_ratio = p_w_var / self.w_var_peak
                    if skipped_count == 0:
                        if w_var_ratio < self.cac_var_start_thr:
                            skip_decision = True
                    else:
                        if self.w_var_mt_count > self.cac_var_mt_count_thr:
                            if w_var_ratio < self.cac_var_mt_thr:
                                skip_decision = True
                        else:
                            if w_var_ratio < self.cac_var_slope_thr:
                                skip_decision = True
                if skip_decision:
                    self.w_var_peak = self.w_var_mt_count = 0
                    # Update the w-var peak of the next target layer
                    self.thresh_stop(group, skipped_count + 1)
        except IndexError:
            pass
        return skip_decision

    def add_braking_distance(self):
        try:
            self.will_stop_layer = self.cac_stop_layer_num[self.skipped_count]
            self.braking_start_lr = self.current_lr
            self.skipped_count += 1
        except IndexError:
            pass

    def braking_distance(self):
        stop_layer_num = -1
        try:
            if self.skip_decision:
                if self.cac_braking_distance > 0:
                    for layer_id in range(self.done_stop_layer + 1,
                                          self.will_stop_layer + 1):
                        if self.braking_count[layer_id] > 0:
                            self.braking_count[layer_id] -= 1
                            if self.braking_count[layer_id] == 0:
                                self.done_stop_layer += 1
                    if self.done_stop_layer == self.will_stop_layer:
                        self.skip_decision = False
                else:
                    self.done_stop_layer = self.will_stop_layer
                    self.skip_decision = False
                stop_layer_num = self.done_stop_layer
        except IndexError:
            pass
        return stop_layer_num

    def stop_layers(self, stop_layer_num):
        if stop_layer_num != -1 and stop_layer_num != self.prev_stop_layer_num:
            for i, p in enumerate(self.org_param_groups):
                if i <= stop_layer_num:
                    p.requires_grad_(False)  # Enabling Gradient-skip
            self.prev_stop_layer_num = stop_layer_num

    def custom_lr(self, layer_id):
        custom_lr_ratio = 1.0
        custom_lr = self.current_lr
        try:
            if (self.cac_braking_distance > 0
                    and layer_id <= self.will_stop_layer):
                custom_lr_ratio = pow((self.braking_count[layer_id] /
                                       self.cac_braking_distance), 2)
                custom_lr = self.braking_start_lr * custom_lr_ratio
        except IndexError:
            pass
        return custom_lr

    def is_conv_layer(self, p):
        return len(p.shape) == 4

    def env_val(self, env_name, default_val=''):
        return self.env_vals(env_name, default_val, is_iterable=False)

    def env_vals(self, env_name, default_val='', is_iterable=True):
        val = ast.literal_eval('[' + os.getenv(env_name, default_val) + ']')
        if is_iterable is False and len(val) == 1:
            return val[0]
        else:
            return val

    def cac_gs_init(self):
        if len(self.param_groups) > 1:
            # When multiple param_group have layers, they cannot be processed
            # in the original layer order and the intended result is not
            # achieved.
            raise ValueError("Not supported for multiple param_groups")

        self.mpi_rank = self.env_val('OMPI_COMM_WORLD_RANK', '-1')
        self.cac_braking_distance = self.env_val('CAC_BRAKING_DISTANCE', '0')
        self.cac_stop_layer_num = self.env_vals('CAC_STOP_LAYER_NUM')
        self.cac_stop_layer_itr = self.env_vals('CAC_STOP_LAYER_ITR')
        self.cac_var_start_itr = self.env_val('CAC_VAR_START_ITR', '5000')
        self.cac_var_start_thr = self.env_val('CAC_VAR_START_THR', '0.95')
        self.cac_var_mt_count_thr = self.env_val('CAC_VAR_MT_COUNT_THR', '5')
        self.cac_var_mt_thr = self.env_val('CAC_VAR_MT_THR', '0.96')
        self.cac_var_slope_thr = self.env_val('CAC_VAR_SLOPE_THR', '0.98')
        self.cac_var_samples = self.env_val('CAC_VAR_SAMPLES', '-1')
        if self.cac_var_samples <= 0:
            # If CAC_VAR_SAMPLES is not defined, the sampling interval is
            # determined as follows:
            #  1. The default sampling interval is 200.
            #  2. If CAC_BRAKING_DISTANCE is defined, the sampling interval
            #     will be set to that value. As a result, Braking Distance will
            #     be executed after detecting the decrease of weight variance.
            if self.cac_braking_distance > 0:
                self.cac_var_samples = self.cac_braking_distance
            else:
                self.cac_var_samples = 200

        self.prev_stop_layer_num = -1
        self.braking_count = {}
        self.current_lr = 0.0
        self.braking_start_lr = 0.0
        self.will_stop_layer = -1
        self.done_stop_layer = -1
        self.skipped_count = 0
        self.skip_decision = False
        self.w_var_peak = 0.0
        self.w_var_mt_count = 0

        if self.is_gs_available():
            for group in self.param_groups:
                layer_num = len(group['params'])
                self.braking_count = [self.cac_braking_distance] * layer_num
                if self.is_manual_stop():
                    for stop_layer_num in self.cac_stop_layer_num:
                        if not 0 <= stop_layer_num < layer_num:
                            raise ValueError("Invalid CAC_STOP_LAYER_NUM: {}".
                                             format(stop_layer_num))
                else:
                    # Set convolution-layer number to cac_stop_layer_num
                    if len(self.cac_stop_layer_num) == 0:
                        for i, p in enumerate(group['params']):
                            if self.is_conv_layer(p):
                                self.cac_stop_layer_num.append(i)

    def cac_gs_execute(self, group=None):
        if group is None:
            raise ValueError("group is required.")
        self.current_lr = group['lr']

        if not self.skip_decision:
            if self.is_manual_stop():
                # Skip decision based on iteration
                self.skip_decision = self.manual_stop()
            else:
                # Skip decision based on weight variance
                self.skip_decision = self.thresh_stop(group)
            if self.skip_decision:
                self.add_braking_distance()

        stop_layer_num = self.braking_distance()
        self.stop_layers(stop_layer_num)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if self.optimizer_itr == 0:
            self.cac_gs_init()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            if self.is_gs_available():
                self.cac_gs_execute(group)

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = \
                            torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                if self.is_gs_available():
                    lr = self.custom_lr(i)
                else:
                    lr = group['lr']
                p.add_(d_p, alpha=-lr)

        self.optimizer_itr += 1

        return loss

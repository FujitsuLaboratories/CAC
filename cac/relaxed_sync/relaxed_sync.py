# relaxed_sync.py COPYRIGHT Fujitsu Limited 2021
import torch
import torch.distributed as dist
from torch.autograd import Variable

from apex.parallel import DistributedDataParallel as DDP
from apex.parallel.distributed import flatten, unflatten, split_half_float_double
from apex.multi_tensor_apply import multi_tensor_applier

import time
import sys

imported_flatten_impl = False


class RelaxedSyncDistributedDataParallel(DDP):
    """
    :class:`apex.parallel.DistributedDataParallel` is a module wrapper that enables
    easy multiprocess distributed data parallel training, similar to ``torch.nn.parallel.DistributedDataParallel``.  Parameters are broadcast across participating processes on initialization, and gradients are
    allreduced and averaged over processes during ``backward()``.

    :class:`DistributedDataParallel` is optimized for use with NCCL.  It achieves high performance by
    overlapping communication with computation during ``backward()`` and bucketing smaller gradient
    transfers to reduce the total number of transfers required.

    :class:`DistributedDataParallel` is designed to work with the upstream launch utility script
    ``torch.distributed.launch`` with ``--nproc_per_node <= number of gpus per node``.
    When used with this launcher, :class:`DistributedDataParallel` assumes 1:1 mapping of processes to GPUs.
    It also assumes that your script calls ``torch.cuda.set_device(args.rank)`` before creating the model.

    https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed shows detailed usage.
    https://github.com/NVIDIA/apex/tree/master/examples/imagenet shows another example
    that combines :class:`DistributedDataParallel` with mixed precision training.

    Args:
        module: Network definition to be run in multi-gpu/distributed mode.
        message_size (int, default=1e7): Minimum number of elements in a communication bucket.
        delay_allreduce (bool, default=False):  Delay all communication to the end of the backward pass.  This disables overlapping communication with computation.
        allreduce_trigger_params (list, optional, default=None):  If supplied, should contain a list of parameters drawn from the model.  Allreduces will be kicked off whenever one of these parameters receives its gradient (as opposed to when a bucket of size message_size is full).  At the end of backward(), a cleanup allreduce to catch any remaining gradients will also be performed automatically.  If allreduce_trigger_params is supplied, the message_size argument will be ignored.
        allreduce_always_fp32 (bool, default=False):  Convert any FP16 gradients to FP32 before allreducing.  This can improve stability for widely scaled-out runs.
        gradient_average (bool, default=True):  Option to toggle whether or not DDP averages the allreduced gradients over processes.  For proper scaling, the default value of True is recommended.
        gradient_predivide_factor (float, default=1.0):  Allows perfoming the average of gradients over processes partially before and partially after the allreduce.  Before allreduce:  ``grads.mul_(1.0/gradient_predivide_factor)``.  After allreduce:  ``grads.mul_(gradient_predivide_factor/world size)``.  This can reduce the stress on the dynamic range of FP16 allreduces for widely scaled-out runs.

    .. warning::
        If ``gradient_average=False``, the pre-allreduce division (``grads.mul_(1.0/gradient_predivide_factor)``) will still be applied, but the post-allreduce gradient averaging (``grads.mul_(gradient_predivide_factor/world size)``) will be omitted.

    """

    def __init__(self,
                 module,
                 message_size=10000000,
                 delay_allreduce=False,
                 shared_param=None,
                 allreduce_trigger_params=None,
                 retain_allreduce_buffers=False,
                 allreduce_always_fp32=False,
                 num_allreduce_streams=1,
                 allreduce_communicators=None,
                 gradient_average=True,
                 gradient_predivide_factor=1.0,
                 gradient_average_split_factor=None,
                 prof=False,
                 relaxed_sync_threshold=3.0,
                 relaxed_sync_mode_threshold=0.8,
                 sleep_process_start_epoch=10, 
                 num_sleep_processes=1,
                 simulate_slow_process=0,
                 sleep_time=0):
        super(RelaxedSyncDistributedDataParallel, self).__init__(module,
                                                                 message_size,
                                                                 delay_allreduce,
                                                                 shared_param,
                                                                 allreduce_trigger_params,
                                                                 retain_allreduce_buffers,
                                                                 allreduce_always_fp32,
                                                                 num_allreduce_streams,
                                                                 allreduce_communicators,
                                                                 gradient_average,
                                                                 gradient_predivide_factor,
                                                                 gradient_average_split_factor,
                                                                 prof)

        self.relaxed_pg = None

        self.relaxed_sync_threshold = relaxed_sync_threshold
        self.relaxed_sync_mode_threshold = relaxed_sync_mode_threshold

        self.comp_time = 0.
        self.comm_time = 0.

        self.start_event = torch.cuda.Event(enable_timing=True, blocking=False)
        self.end_event = torch.cuda.Event(enable_timing=True, blocking=False)

        self.old_pgsize = dist.get_world_size()

        self.simulate_slow_process = simulate_slow_process
        self.sleep_process_start_epoch = sleep_process_start_epoch
        self.num_sleep_processes = num_sleep_processes
        self.sleep_time = sleep_time
        self.print_once = 0

        self.relaxed_process_ids = []
        self.epoch = 0

        self.fwd_cnt = 0

    def set_relaxed_pg(self, epoch, min_num_processes=1, master_skip=False, new_pg=True):
        group = self.relaxed_pg if self.relaxed_pg is not None else dist.group.WORLD
        self.epoch = epoch

        if self.simulate_slow_process == 1 and self.sleep_time == 0 and self.fwd_cnt > 0:
            self.sleep_time = self.comp_time/self.fwd_cnt * 30
            self.fwd_cnt = 0

        # calculate average computation time in the group
        if dist.get_world_size(group) == 1: 
                return 
        if dist.get_world_size(group) == 2: # assume smallest group size is 2
            if dist.get_rank(group) >= 0:
                comp_time_list = [torch.cuda.FloatTensor([0.]) for _ in range(dist.get_world_size(group))]
                comp_time = torch.cuda.FloatTensor([self.comp_time])
                dist.all_gather(comp_time_list, comp_time, group)
                average_time = torch.min(torch.stack(comp_time_list), dim=0).values
        elif dist.get_world_size(group) > 2:
            if dist.get_rank(group) >= 0:
                comp_time_list = [torch.cuda.FloatTensor([0.]) for _ in range(dist.get_world_size(group))]
                comp_time = torch.cuda.FloatTensor([self.comp_time])
                dist.all_gather(comp_time_list, comp_time, group)
                #average_time = sum(comp_time_list) / dist.get_world_size(group)
                average_time = torch.median(torch.stack(comp_time_list), dim=0).values

        #if dist.get_rank(group) == 0:
        #    print("global_rank:", dist.get_rank(), ", rank_in_group:", dist.get_rank(group),
        #          ", set_relaxed_pg: average_time:", average_time.item(), ", my_time:", self.comp_time, ", comp_time_list:", comp_time_list)

        # get process ids which satisfies criteria with threshold
        # do not skip master if master_skip is false
        is_not_slow = 1 if self.relaxed_sync_threshold < 0.0 \
            or dist.get_rank(group) >= 0 and average_time.item() > 0.0 \
            and self.comp_time <= average_time.item() * self.relaxed_sync_threshold \
            else 0
        if average_time.item() == 0.0 and dist.get_rank(group) >=0:
           is_not_slow = 1
        if not master_skip and dist.get_rank(group) == 0:
            is_not_slow = 1
        if dist.get_rank(group) >= 0 and not is_not_slow and average_time.item() > 0.0:
            print("slow_process: global_rank:", dist.get_rank(), ", average_time:", average_time.item(), ", my_time:", self.comp_time)
        is_relaxed_process = torch.cuda.IntTensor([is_not_slow])
        is_relaxed_process_list = [torch.cuda.IntTensor([0]) for _ in range(dist.get_world_size(group))]
        # TODO: recover the relaxed processes when the set_relaxed_pg method is called more than twice
        # gather only from the processes in the relaxed process group
        # NCCL does not support gather
        # dist.gather(is_relaxed_process, is_relaxed_process_list, dst=dist.get_rank(group), group=group)
        dist.all_gather(is_relaxed_process_list, is_relaxed_process, group)

        relaxed_process_ids = []
        for rank in range(dist.get_world_size(group)):
            if is_relaxed_process_list[rank].item() > 0:
                relaxed_process_ids.append(self.relaxed_process_ids[rank] if self.relaxed_pg is not None else rank)

        # broadcast to all the processes in the default process group
        # do not use the gather->broadcast but use all_gather because NCCL does not support gather
        # relaxed_process_ids_tensor = torch.cuda.IntTensor(relaxed_process_ids)
        # dist.broadcast(relaxed_process_ids_tensor, dist.get_rank())

        if len(relaxed_process_ids) >= min_num_processes and len(relaxed_process_ids) < dist.get_world_size(group):
            self.relaxed_process_ids = relaxed_process_ids
            if dist.get_rank(group) == 0:
                print("create new group: comm_size:", len(relaxed_process_ids), ", relaxed_process_ids:", self.relaxed_process_ids)
            if new_pg:
                if self.relaxed_pg is not None:
                    dist.destroy_process_group(self.relaxed_pg)
                self.relaxed_pg = dist.new_group(relaxed_process_ids)

        self.comp_time = 0

    def calc_prec(self, loss_data, prec1, prec5):
        group = self.relaxed_pg if self.relaxed_pg is not None else dist.group.WORLD
        dist.all_reduce(loss_data, group=group)
        dist.all_reduce(prec1, group=group)
        dist.all_reduce(prec5, group=group)
        reduced_loss = loss_data / dist.get_world_size(group)
        prec1 /= torch.distributed.get_world_size(group)
        prec5 /= torch.distributed.get_world_size(group)
        return reduced_loss, prec1, prec5

    def DistributedSampler(self, val_dataset):
        if self.relaxed_pg is not None:
            return torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                                   rank=dist.get_rank(),
                                                                   num_replicas=dist.get_world_size(self.relaxed_pg))
        else:
            return torch.utils.data.distributed.DistributedSampler(val_dataset)

    def calc_reduced_tensor(self, tensor_data):
        group = self.relaxed_pg if self.relaxed_pg is not None else dist.group.WORLD
        dist.all_reduce(tensor_data, group=group)
        reduced_tensor = tensor_data / dist.get_world_size(group)
        return reduced_tensor

    def is_group_size_changed(self):
        group = self.relaxed_pg if self.relaxed_pg is not None else dist.group.WORLD
        pgsize = dist.get_world_size(group)
        if self.old_pgsize != pgsize:
            self.old_pgsize = pgsize
            return True
        else:
            return False

    def create_hooks(self):
        # Fallback hook that's only called at the end of backward.
        # Used if you deliberately want to delay allreduces to the end, or to refresh the
        # bucket structure that will be used to overlap communication with computation in later
        # iterations.

        def allreduce_params():
            # Bucket record refresh
            if not self.delay_allreduce:
                if self.needs_refresh:
                    self.sync_bucket_structure()

                    self.needs_refresh = False

            self.allreduce_fallback()

        def overlapping_backward_epilogue():
            for stream, event in zip(self.bucket_streams, self.bucket_events):
                stream.record_event(event)
                torch.cuda.current_stream().wait_event(event)

            # Sanity checks that all the buckets were kicked off
            if self.next_bucket != self.num_buckets:
                raise RuntimeError("In epilogue, next_bucket ({}) != num_buckets ({}).  ".format(
                                   self.next_bucket, self.num_buckets),
                                   "This probably indicates some buckets were not allreduced.")

            for actual, expected in zip(self.buckets_ready_size, self.bucket_sizes):
                if actual != expected:
                    raise RuntimeError("Some param buckets were not allreduced.")

        self.grad_accs = []
        for param in self.module.parameters():
            if dist.get_rank() == 0:
                print("param: ", param.requires_grad, ", ", type(param.data), ", ", param.dtype, ", ", param.size())
            if param.requires_grad:
                def wrapper(param):
                    param_tmp = param.expand_as(param)
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]

                    def allreduce_hook(*unused):
                        if self.prof:
                            torch.cuda.nvtx.range_push("allreduce_hook")

                        if not self._disable_allreduce:
                            if self.delay_allreduce or self.needs_refresh:
                                # TODO:  How do we want to handle multiple backward passes between
                                # each forward, e.g., backward passes with retain_graph=True?
                                # needs_refresh and callback_queued are both vulnerable states.
                                if not self.delay_allreduce and self.needs_refresh:
                                    # Use the backward pass to build the bucket structure on the fly.
                                    active_i = self.param_id_to_active_i[id(param)]

                                    # Float, half, and double tensors are grouped into buckets separately.
                                    current_type = self.param_type_to_tmp_i[param.type()]

                                    self.tmp_buckets[current_type].append(active_i)

                                    ship_tmp_bucket = False
                                    if self.custom_allreduce_triggers:
                                        if id(param) in self.allreduce_trigger_params:
                                            ship_tmp_bucket = True
                                    else:
                                        self.tmp_numels[current_type] += param.numel()
                                        if self.tmp_numels[current_type] >= self.message_size:
                                            ship_tmp_bucket = True

                                    # To consider:  If custom_allreduce_triggers are in use, ship all
                                    # tmp_buckets, not just tmp_buckets[current_type].
                                    if ship_tmp_bucket:
                                        self.active_i_buckets.append(self.tmp_buckets[current_type])
                                        self.tmp_buckets[current_type] = []
                                        self.tmp_numels[current_type] = 0

                                if not self.callback_queued:
                                    Variable._execution_engine.queue_callback(allreduce_params)
                                    self.callback_queued = True
                            else:
                                if not self.callback_queued:
                                    Variable._execution_engine.queue_callback(overlapping_backward_epilogue)
                                    self.callback_queued = True

                                self.comm_ready_buckets(param)

                        if self.prof:
                            torch.cuda.nvtx.range_pop()

                    grad_acc.register_hook(allreduce_hook)
                    self.grad_accs.append(grad_acc)

                wrapper(param)

    def allreduce_bucket(self, bucket, bucket_idx, force_default_stream):

        tensor = flatten(bucket)

        if force_default_stream:
            bucket_stream = self.main_stream
        else:
            bucket_stream = self._stream_this_bucket(bucket_idx)
            bucket_event = self._event_this_bucket(bucket_idx)
            torch.cuda.current_stream().record_event(bucket_event)
            bucket_stream.wait_event(bucket_event)

        with torch.cuda.stream(bucket_stream):
            # self.main_stream.wait_stream(torch.cuda.current_stream())
            # torch.cuda.synchronize()

            tensor_to_allreduce = tensor

            if self.allreduce_always_fp32:
                tensor_to_allreduce = tensor.float()

            if self.gradient_predivide_factor != 1.0:
                tensor_to_allreduce.mul_(1./self.gradient_predivide_factor)

            if self.allreduce_different_streams and not force_default_stream:
                dist.all_reduce(tensor_to_allreduce, group=self.bucket_pgs[bucket_idx % self.num_allreduce_streams])
            else:
                if self.relaxed_pg is not None:
                    dist.all_reduce(tensor_to_allreduce, group=self.relaxed_pg)
                else:
                    dist.all_reduce(tensor_to_allreduce)

            if self.gradient_average:
                group = self.relaxed_pg if self.relaxed_pg is not None else dist.group.WORLD
                self.world_size = dist.get_world_size(group)
                tensor_to_allreduce.mul_(self.gradient_predivide_factor/self.world_size)

            if self.allreduce_always_fp32 and tensor is not tensor_to_allreduce:
                tensor.copy_(tensor_to_allreduce)

            if not self.retain_allreduce_buffers:
                if multi_tensor_applier.available:
                    multi_tensor_applier(
                        self.multi_tensor_scale,
                        self._overflow_buf,
                        [unflatten(tensor, bucket), bucket],
                        1.0)
                else:
                    for buf, synced in zip(bucket, unflatten(tensor, bucket)):
                        buf.copy_(synced)
            # I think we actually do need this here.  After allreduce_bucket returns, tensor will
            # eventually go out of scope and die, at which point it could otherwise be freed for
            # further reuse by the main stream while the allreduce/div/unflatten are underway in bucket_stream.
            tensor.record_stream(bucket_stream)

        return tensor

    def allreduce_fallback(self):
        self.start_event.record()
        for stream, event in zip(self.bucket_streams, self.bucket_events):
            stream.record_event(event)
            torch.cuda.current_stream().wait_event(event)

        if self.retain_allreduce_buffers:
            grads = [param.grad for param in self.module.parameters() if param.grad is not None]
        else:
            grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]

        split_buckets = split_half_float_double(grads)

        # If retain_allreduce_buffers is True and delay_allreduce is False,
        # this will only be done during the first backward pass, ignored by the
        # training script, and overwritten in the next forward pass.  So it's harmless.
        if self.retain_allreduce_buffers:
            self.allreduce_buffers = [None for _ in range(len(split_buckets))]

        for i, bucket in enumerate(split_buckets):
            self.allreduce_maybe_retain(bucket, i, force_default_stream=True)

        self.end_event.record()

        self.end_event.synchronize()
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        self.comm_time = 0.001 * elapsed_time_ms

    def forward(self, *inputs, **kwargs):
        start_time_Throughput = time.time()
        result = self.module(*inputs, **kwargs)

        if self.prof:
            torch.cuda.nvtx.range_push("forward pass DDP logic")

        if self.simulate_slow_process == 1:
            if torch.distributed.get_rank() >= torch.distributed.get_world_size() - self.num_sleep_processes and self.epoch >= self.sleep_process_start_epoch:
                if self.print_once == 0:
                    print("simulating slow process")
                    self.print_once = 1
                time.sleep(self.sleep_time)

        group = self.relaxed_pg if self.relaxed_pg is not None else dist.group.WORLD
        if dist.get_rank(group) == -1:
            while True:
                time.sleep(1)
                dist.barrier()
                sys.exit()

        if not self._disable_allreduce:
            if not self.delay_allreduce:
                param_list = [param for param in self.module.parameters() if param.requires_grad]

                # Conditions under which to refresh self.record
                # Forward has the authority to set needs_refresh to True, but only allreduce_params
                # in backward has the authority to set needs_refresh to False.
                # Parentheses are not necessary for correct order of operations, but make the intent clearer.
                if ((not self.active_params) or
                    (len(param_list) != len(self.active_params)) or
                        any([param1 is not param2 for param1, param2 in zip(param_list, self.active_params)])):
                    self.needs_refresh = True

                if self.needs_refresh:
                    self.active_i_buckets = []
                    self.buckets = []
                    self.tmp_buckets = [[], [], []]  # [running half, float, double buckets]
                    self.tmp_numels = [0, 0, 0]
                    self.bucket_sizes = []
                    self.param_id_to_active_i = {id(param): i for i, param in enumerate(param_list)}
                    self.param_id_to_bucket = {}
                    self.bucket_pgs = []
                    self.bucket_streams = []
                    self.bucket_events = []
                else:
                    # self.buckets = [[None for _ in range(self.bucket_sizes[i])]
                    #                 for i in range(self.num_buckets)]
                    if not self.buckets:
                        self.buckets = [[None for _ in range(self.bucket_sizes[i])]
                                        for i in range(self.num_buckets)]
                    else:
                        assert len(self.buckets) == self.num_buckets, "len(buckets) = {}, expected {}".format(
                            len(self.buckets), self.num_buckets)
                        for b, bucket in enumerate(self.buckets):
                            assert len(bucket) == self.bucket_sizes[b], "len(buckets[{}]) = {}, expected {})".format(
                                b, len(self.buckets[b]), self.bucket_sizes[b])
                            for i in range(len(bucket)):
                                bucket[i] = None

                    if self.allreduce_communicators:
                        self.bucket_pgs = self.allreduce_communicators[0]
                        self.bucket_streams = self.allreduce_communicators[1]
                        self.bucket_events = [torch.cuda.Event(enable_timing=False,
                                                               blocking=False) for _ in range(self.num_allreduce_streams)]
                    else:
                        if self.allreduce_different_streams:
                            if not self.bucket_pgs:
                                self.bucket_pgs = [dist.new_group() for _ in range(self.num_allreduce_streams)]
                                for i, bg in enumerate(self.bucket_pgs):
                                    print("rank {} created group {} with backend {}".format(
                                          dist.get_rank(), i, dist.get_backend(bg)))
                        if self.allreduce_different_streams:
                            if not self.bucket_streams:
                                self.bucket_streams = [torch.cuda.Stream() for _ in range(self.num_allreduce_streams)]
                                self.bucket_events = [torch.cuda.Event(enable_timing=False,
                                                      blocking=False) for _ in range(self.num_allreduce_streams)]
                        else:
                            if not self.bucket_streams:
                                self.bucket_streams = [torch.cuda.Stream()]
                                self.bucket_events = [torch.cuda.Event(enable_timing=False, blocking=False)]

                    self.buckets_ready_size = [0 for i in range(self.num_buckets)]
                    if(self.retain_allreduce_buffers):
                        self.allreduce_buffers = [None for _ in range(self.num_buckets)]
                    self.next_bucket = 0
                    self.ready_buckets_not_reduced = set()

                self.active_params = param_list

            self.callback_queued = False

        if self.prof:
            torch.cuda.nvtx.range_pop()

        self.comp_time += time.time() - start_time_Throughput
        if self.simulate_slow_process == 1:
            self.fwd_cnt += 1
        return result

    def TrainDataLoader(self,
                        dataset,
                        batch_size=1,
                        shuffle=False,
                        sampler=None,
                        batch_sampler=None,
                        num_workers=0,
                        collate_fn=None,
                        pin_memory=False,
                        drop_last=False,
                        timeout=0,
                        worker_init_fn=None,
                        prefetch_factor=2,
                        persistent_workers=False):

        self.train_dataset = dataset
        self.train_batch_size = batch_size
        self.train_shuffle = shuffle
        self.train_sampler = sampler
        self.train_batch_sampler = batch_sampler
        self.train_num_workers = num_workers
        self.train_collate_fn = collate_fn
        self.train_pin_memory = pin_memory
        self.train_drop_last = drop_last
        self.train_timeout = timeout
        self.train_worker_init_fn = worker_init_fn
        self.train_prefetch_factor = prefetch_factor
        self.train_persistent_workers = persistent_workers

        return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                shuffle=self.train_shuffle,
                sampler=self.train_sampler,
                batch_sampler=self.train_batch_sampler,
                num_workers=self.train_num_workers,
                collate_fn=self.train_collate_fn,
                pin_memory=self.train_pin_memory,
                drop_last=self.train_drop_last,
                timeout=self.train_timeout,
                worker_init_fn=self.train_worker_init_fn
                )

    def ValDataLoader(self,
                      dataset,
                      batch_size=1,
                      shuffle=False,
                      sampler=None,
                      batch_sampler=None,
                      num_workers=0,
                      collate_fn=None,
                      pin_memory=False,
                      drop_last=False,
                      timeout=0,
                      worker_init_fn=None,
                      prefetch_factor=2,
                      persistent_workers=False):

        self.val_dataset = dataset
        self.val_batch_size = batch_size
        self.val_shuffle = shuffle
        self.val_sampler = sampler
        self.val_batch_sampler = batch_sampler
        self.val_num_workers = num_workers
        self.val_collate_fn = collate_fn
        self.val_pin_memory = pin_memory
        self.val_drop_last = drop_last
        self.val_timeout = timeout
        self.val_worker_init_fn = worker_init_fn
        self.val_prefetch_factor = prefetch_factor
        self.val_persistent_workers = persistent_workers

        return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                shuffle=self.val_shuffle,
                sampler=self.val_sampler,
                batch_sampler=self.val_batch_sampler,
                num_workers=self.val_num_workers,
                collate_fn=self.val_collate_fn,
                pin_memory=self.val_pin_memory,
                drop_last=self.val_drop_last,
                timeout=self.val_timeout,
                worker_init_fn=self.val_worker_init_fn
                )

    def rearrange_data_loaders(self, train_loader, val_loader):
        # procs_remained = len(self.relaxed_process_ids) / torch.distributed.get_world_size()
        group = self.relaxed_pg if self.relaxed_pg is not None else dist.group.WORLD
        relaxed_size = torch.distributed.get_world_size(group)
        init_world_size = torch.distributed.get_world_size()

        if(self.is_group_size_changed() is True) and (torch.distributed.get_rank(group) != -1):
            if relaxed_size/init_world_size < self.relaxed_sync_mode_threshold:
                if torch.distributed.get_rank() == 0:
                    print("re-create train_loader")

                train_loader = self.create_new_train_loader()

            if torch.distributed.get_rank() == 0:
                print("re-create val_loader")
            val_loader = self.create_new_val_loader()

        return train_loader, val_loader

    def create_new_train_loader(self):
        _rank = torch.distributed.get_rank(self.relaxed_pg)
        _world_size = torch.distributed.get_world_size(self.relaxed_pg)
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, rank=_rank, num_replicas=_world_size)

        return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                shuffle=self.train_shuffle,
                sampler=train_sampler,
                batch_sampler=self.train_batch_sampler,
                num_workers=self.train_num_workers,
                collate_fn=self.train_collate_fn,
                pin_memory=self.train_pin_memory,
                drop_last=self.train_drop_last,
                timeout=self.train_timeout,
                worker_init_fn=self.train_worker_init_fn
                )

    def create_new_val_loader(self):
        _rank = torch.distributed.get_rank(self.relaxed_pg)
        _world_size = torch.distributed.get_world_size(self.relaxed_pg)
        val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, rank=_rank, num_replicas=_world_size)

        return torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                shuffle=self.val_shuffle,
                sampler=val_sampler,
                batch_sampler=self.val_batch_sampler,
                num_workers=self.val_num_workers,
                collate_fn=self.val_collate_fn,
                pin_memory=self.val_pin_memory,
                drop_last=self.val_drop_last,
                timeout=self.val_timeout,
                worker_init_fn=self.val_worker_init_fn
                )

    def adjust_lr_by_procs(self, lr):
        group = self.relaxed_pg if self.relaxed_pg is not None else dist.group.WORLD
        relaxed_size = torch.distributed.get_world_size(group)
        init_world_size = torch.distributed.get_world_size()
        lr_tmp = lr

        if torch.distributed.get_rank(group) != -1:
            if relaxed_size/init_world_size < self.relaxed_sync_mode_threshold:
                remain_procs = relaxed_size/init_world_size

                lr_tmp = lr*(remain_procs)

        return lr_tmp

    def finalize(self):
        dist.barrier()



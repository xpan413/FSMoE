# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Any

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.fusions.fused_bias_gelu import bias_gelu_back
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.callback import CallbackBase
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.moe_utils import (
    permute,
    sort_chunks_by_idxs,
    topk_softmax_with_capacity,
    unpermute_with_padded_tokens,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        if self.config.moe_extended_tp:
            self.num_local_experts = self.config.num_moe_experts
            local_expert_indices_offset = 0
        else:
            assert self.config.num_moe_experts % self.expert_parallel_size == 0
            self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
            local_expert_indices_offset = (
                parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
            )

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class LegacyMoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        if self.config.moe_grouped_gemm:
            if isinstance(self.submodules, MLPSubmodules):
                self.experts = TEGroupedMLP(self.num_local_experts, self.config, self.submodules)
            else:
                self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                self.comm_stream,
                self.comm_event_lst,
                config=self.config,
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )
        self.moe_layer_recompute = config.moe_layer_recompute

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            probs, indices = self.router(hidden_states)
            (dispatched_input_lst, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, indices
            )
            expert_output_lst = []
            for i, dispatched_input in enumerate(dispatched_input_lst):
                torch.cuda.current_stream().wait_event(self.comm_event_lst[i])
                # print(dispatched_input.shape, tokens_per_expert)
                expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
                expert_output_lst.append(expert_output)
                self.comm_event_lst[i].record()
            output, mlp_bias = self.token_dispatcher.token_unpermutation(
                expert_output_lst, mlp_bias
            )
            return output, mlp_bias

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias


def _assert_config_attribute(
    config: TransformerConfig, attribute: str, expected_value: Any
) -> None:
    """Asserts that a given attribute of the config matches the expected value."""
    actual_value = getattr(config, attribute, None)
    assert actual_value == expected_value, f"We don't test {attribute} yet."


class _HandcraftBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, output, moe):
        ctx.moe = moe
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # print(grad_output.shape)
        return ctx.moe._handcraft_backward(grad_output), None, None


class MoELayer(BaseMoELayer):
    """FsMoE Layer**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules = None,
        layer_number: int = None,
        callbacks: list[CallbackBase] = [],
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)
        _assert_config_attribute(config, "moe_grouped_gemm", False)
        assert isinstance(self.submodules, MLPSubmodules)
        self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)
        _assert_config_attribute(config, "moe_token_dispatcher_type", "alltoall")
        # self.token_dispatcher = MoEAlltoAllTokenDispatcher(
        #     self.num_local_experts, self.local_expert_indices, config=self.config
        # )
        self.moe_layer_recompute = config.moe_layer_recompute
        self.capacity_factor = self.config.moe_expert_capacity_factor
        assert self.capacity_factor is not None, "We don't test self.capacity_factor is None yet."

        self.callbacks = callbacks
        self.ep_size = config.expert_model_parallel_size
        self.tp_size = config.tensor_model_parallel_size
        self.topk = self.config.moe_router_topk
        self.pipeline_degree = self.config.pipeline_degree
        self.bp_pipeline_degree = self.config.bp_pipeline_degree
        self.inter_stream = torch.cuda.Stream()  # We add another comm to use communication
        self.intra_stream = torch.cuda.Stream()
        self.comm_event_lst = [
            torch.cuda.Event() for _ in range(max(self.pipeline_degree, self.bp_pipeline_degree))
        ]
        input_chunk_idxs = torch.arange(self.config.num_moe_experts * self.tp_size)
        # [num_local_experts, tp_size * ep_size]. Sort the input chunks by local experts.
        self.sort_input_by_local_experts = (
            input_chunk_idxs.reshape(-1, self.num_local_experts).T.ravel().tolist()
        )
        self.restore_output_by_local_experts = (
            input_chunk_idxs.reshape(self.num_local_experts, -1).T.ravel().tolist()
        )

        def find_all_leaf_module(module: torch.nn.Module):
            if not list(module.children()):
                print(module)
                _forward_call = module.forward
                pipeline_degree = self.pipeline_degree
                module.start = pipeline_degree

                def forward(self, input_, *args, **kwargs):
                    if self.start == pipeline_degree:
                        self._input_buffer = torch.empty(
                            (pipeline_degree * input_.numel(),),
                            dtype=input_.dtype,
                            device=input_.device,
                        )
                        self.start = 0
                        self.length = input_.numel()
                    buf = torch.narrow(self._input_buffer, 0, self.start * self.length, self.length)
                    buf.copy_(torch.flatten(input_))
                    self.start += 1
                    return _forward_call(input_, *args, **kwargs)

                import types

                module.forward = types.MethodType(forward, module)

            for child in module.children():
                find_all_leaf_module(child)

        find_all_leaf_module(self.experts)

    def do_router(self, batch: dict) -> None:
        self.router(batch)

    def do_order(self, batch: dict) -> None:

        batch["probs"], batch["indices"], _ = topk_softmax_with_capacity(
            batch["logits"], self.topk, self.capacity_factor, True
        )
        batch["iorder_data_shape"] = batch["data"].shape
        batch["data"] = batch["data"].index_select(dim=0, index=batch["indices"].view(-1))
        # print(batch["data"].shape)

    def do_dispatch(self, batch: dict) -> None:
        r"""
        The collective communication to dispatch the data.
        """
        with torch.cuda.stream(self.inter_stream):
            batch["event"].wait()
            batch["data"] = tensor_parallel.all_to_all(
                parallel_state.get_expert_model_parallel_group(), batch["data"]
            )
            batch["event"].record()
        with torch.cuda.stream(self.intra_stream):
            batch["event"].wait()
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                batch["data"] = gather_from_sequence_parallel_region(batch["data"])
            batch["event"].record()

    def do_experts(self, batch: dict) -> None:
        r"""
        The computation task for experts.
        """
        batch["event"].wait()
        if self.num_local_experts > 1:
            batch["data"] = sort_chunks_by_idxs(
                batch["data"],
                batch["num_global_tokens_per_local_expert_cpu"].ravel(),
                self.sort_input_by_local_experts,
            )
        batch["data"], _ = self.experts(batch["data"], batch["tokens_per_expert"])
        if self.num_local_experts > 1:
            batch["data"] = sort_chunks_by_idxs(
                batch["data"],
                batch["num_global_tokens_per_local_expert_cpu"].T.ravel(),
                self.restore_output_by_local_experts,
            )
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        dim_size = list(batch["data"].size())
        dim_size[0] = dim_size[0] // world_size
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            batch["combine_output"] = torch.empty(
                dim_size, dtype=batch["data"].dtype, device=torch.cuda.current_device()
            )
        batch["data"] = batch["data"].contiguous()

        batch["event"].record()

    def do_bp_experts(self, batch: dict) -> None:
        r"""
        The computation task for experts.
        """
        batch["event"].wait()
        if self.num_local_experts > 1:
            batch["data"] = sort_chunks_by_idxs(
                batch["data"],
                batch["num_global_tokens_per_local_expert_cpu"].ravel(),
                self.sort_input_by_local_experts,
            )
        tokens_per_expert = batch["tokens_per_expert"].tolist()
        tokens_list = torch.split(batch["data"], tokens_per_expert)
        batch_size = tokens_list[0].shape[0]
        output_local_list = []
        for expert, tokens in zip(self.experts.local_experts, tokens_list):
            length = expert.linear_fc2._input_buffer.numel() // self.bp_pipeline_degree
            input = torch.narrow(expert.linear_fc2._input_buffer, 0, batch["idx"] * length, length)
            output = tokens.matmul(expert.linear_fc2.weight)
            grad_weight = tokens.t().matmul(input.view(batch_size, -1))
            expert.linear_fc2.weight.main_grad += grad_weight
            # output = expert.activation_func.backward(output)
            # output = (output > 0).to(output.dtype)
            output[output < 0] = 0
            output_local_list.append(output)

        output2_local_list = []
        for expert, tokens in zip(self.experts.local_experts, output_local_list):
            length = expert.linear_fc1._input_buffer.numel() // self.bp_pipeline_degree
            input = torch.narrow(expert.linear_fc1._input_buffer, 0, batch["idx"] * length, length)
            output = tokens.matmul(expert.linear_fc1.weight)
            grad_weight = tokens.t().matmul(input.view(batch_size, -1))
            expert.linear_fc1.weight.main_grad += grad_weight
            # output, output_bias = expert(tokens)
            output2_local_list.append(output)
        batch["data"] = torch.cat(output2_local_list, dim=0)
        # batch["data"], _ = self.experts(batch["data"], batch["tokens_per_expert"])
        if self.num_local_experts > 1:
            batch["data"] = sort_chunks_by_idxs(
                batch["data"],
                batch["num_global_tokens_per_local_expert_cpu"].T.ravel(),
                self.restore_output_by_local_experts,
            )

        world_size = parallel_state.get_tensor_model_parallel_world_size()
        dim_size = list(batch["data"].size())
        dim_size[0] = dim_size[0] // world_size
        if parallel_state.get_tensor_model_parallel_world_size() > 1:
            batch["combine_output"] = torch.empty(
                dim_size, dtype=batch["data"].dtype, device=torch.cuda.current_device()
            )
        batch["event"].record()

    def do_combine(self, batch: dict) -> None:
        r"""
        The collective communication to do_combine the data.
        """
        with torch.cuda.stream(self.intra_stream):
            batch["event"].wait()
            if parallel_state.get_tensor_model_parallel_world_size() > 1:
                torch.distributed._reduce_scatter_base(
                    batch["combine_output"],
                    batch["data"],
                    group=parallel_state.get_tensor_model_parallel_group(),
                )
                batch["data"] = batch["combine_output"]
            batch["event"].record()
        with torch.cuda.stream(self.inter_stream):
            batch["event"].wait()
            batch["data"] = tensor_parallel.all_to_all(
                parallel_state.get_expert_model_parallel_group(), batch["data"]
            )
            batch["event"].record()

    def do_iorder(self, batch: dict) -> None:
        batch["data"] = unpermute_with_padded_tokens(
            batch["data"], batch["indices"], batch["probs"], batch["iorder_data_shape"]
        )

    def before_moe_start_hook(self, batch: dict) -> None:
        for callback in self.callbacks:
            callback.before_moe_start_hook(self, batch)

    def before_dispatch_hook(self, batch: dict) -> None:
        for callback in self.callbacks:
            callback.before_dispatch_hook(self, batch)

    def after_dispatch_hook(self, batch: dict) -> None:
        for callback in self.callbacks:
            callback.after_dispatch_hook(self, batch)

    def before_combine_hook(self, batch: dict) -> None:
        for callback in self.callbacks:
            callback.before_combine_hook(self, batch)

    def after_combine_hook(self, batch: dict) -> None:
        for callback in self.callbacks:
            callback.after_combine_hook(self, batch)

    def before_moe_end_hook(self, batch: dict) -> None:
        for callback in self.callbacks:
            callback.before_moe_end_hook(self, batch)

    def _handcraft_backward(self, grad_input):
        batch = {"data": grad_input}
        # print(grad_input.shape)
        hidden_dim = batch["data"].shape[-1]
        batch["data"] = batch["data"].view(self.config.num_moe_experts, -1, hidden_dim)
        pipelined_data = torch.tensor_split(batch["data"], self.bp_pipeline_degree, 1)

        pipelined_tokens_per_expert = torch.full(
            (self.num_local_experts,),
            batch["data"].shape[1] * self.tp_size * self.ep_size // self.bp_pipeline_degree,
            dtype=torch.long,
        )

        pipelined_num_global_tokens_per_local_expert_cpu = torch.full(
            (self.config.num_moe_experts * self.tp_size,),
            batch["data"].shape[1] // self.bp_pipeline_degree,
            dtype=torch.long,
        )

        micro_batch = [dict() for _ in range(self.bp_pipeline_degree)]
        for idx, (data, event) in enumerate(zip(pipelined_data, self.comm_event_lst)):
            _micro_shape = data.shape
            micro_batch[idx]["data"] = data.reshape(-1, _micro_shape[-1])
            micro_batch[idx]["tokens_per_expert"] = pipelined_tokens_per_expert
            micro_batch[idx][
                "num_global_tokens_per_local_expert_cpu"
            ] = pipelined_num_global_tokens_per_local_expert_cpu
            micro_batch[idx]["event"] = event
            micro_batch[idx]["idx"] = idx
            event.record()

        for _batch in micro_batch:
            self.before_dispatch_hook(_batch)
            self.do_dispatch(_batch)
            self.after_dispatch_hook(_batch)
        for _batch in micro_batch:
            self.do_bp_experts(_batch)
            self.before_combine_hook(_batch)
            self.do_combine(_batch)
            self.after_combine_hook(_batch)

        output_lst = []
        for _batch in micro_batch:
            _batch["event"].wait()
            output_lst.append(_batch["data"].view(_micro_shape))
        batch["data"] = torch.cat(output_lst, dim=1).reshape(-1, _micro_shape[-1])

        for idx in range(self.bp_pipeline_degree - 1, -1, -1):
            del micro_batch[idx]
        del micro_batch

        grad_output = batch["data"]
        del batch
        return grad_output

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            # probs, indices = self.router(hidden_states)
            # (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
            #     hidden_states, probs, indices
            # )
            # expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            # output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            # return output, mlp_bias
            batch = {"original_data": hidden_states}
            batch["data"] = batch["original_data"].view(-1, hidden_states.shape[-1])
            self.before_moe_start_hook(batch)
            self.do_router(batch)
            self.do_order(batch)
            batch["grad_data"] = batch["data"]
            with torch.no_grad():
                pipelined_data = torch.tensor_split(
                    batch["data"].view(*batch["indices"].shape, -1), self.pipeline_degree, 1
                )
                pipelined_tokens_per_expert = torch.full(
                    (self.num_local_experts,),
                    batch["indices"].shape[1] * self.tp_size * self.ep_size // self.pipeline_degree,
                    dtype=torch.long,
                )

                pipelined_num_global_tokens_per_local_expert_cpu = torch.full(
                    (self.config.num_moe_experts * self.tp_size,),
                    batch["indices"].shape[1] // self.pipeline_degree,
                    dtype=torch.long,
                )

                micro_batch = [dict() for _ in range(self.pipeline_degree)]
                for idx, (data, event) in enumerate(zip(pipelined_data, self.comm_event_lst)):
                    _micro_shape = data.shape
                    micro_batch[idx]["data"] = data.reshape(-1, _micro_shape[-1])
                    micro_batch[idx]["tokens_per_expert"] = pipelined_tokens_per_expert
                    micro_batch[idx][
                        "num_global_tokens_per_local_expert_cpu"
                    ] = pipelined_num_global_tokens_per_local_expert_cpu
                    micro_batch[idx]["event"] = event
                    event.record()

                for _batch in micro_batch:
                    self.before_dispatch_hook(_batch)
                    self.do_dispatch(_batch)
                    self.after_dispatch_hook(_batch)
                for _batch in micro_batch:
                    self.do_experts(_batch)
                    self.before_combine_hook(_batch)
                    self.do_combine(_batch)
                    self.after_combine_hook(_batch)

                output_lst = []
                for _batch in micro_batch:
                    _batch["event"].wait()
                    output_lst.append(_batch["data"].view(_micro_shape))
                batch["data"] = torch.cat(output_lst, dim=1).reshape(-1, _micro_shape[-1])
            batch["data"] = batch["data"].detach()
            batch["data"].requires_grad = True
            batch["data"] = _HandcraftBackward.apply(batch["grad_data"], batch["data"], self)
            for idx in range(self.pipeline_degree - 1, -1, -1):
                del micro_batch[idx]
            del micro_batch
            self.do_iorder(batch)
            self.before_moe_end_hook(batch)
            # print(batch)
            # batch["data"].detach()
            # batch["data"].requires_grad = True
            # batch["data"] = _HandcraftBackward.apply(batch["grad_data"], batch["data"], self)
            hidden_states = batch["data"].reshape_as(batch["original_data"])
            del batch
            return hidden_states, None

        if self.moe_layer_recompute:
            output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

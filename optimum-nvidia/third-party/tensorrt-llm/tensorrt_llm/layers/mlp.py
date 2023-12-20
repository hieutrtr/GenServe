# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np

from .._utils import trt_dtype_to_np
from ..functional import ACT2FN
from ..module import Module
from ..quantization import QuantMode
from ..quantization.layers import FP8Linear, FP8RowLinear
from .linear import ColumnLinear, RowLinear


class MLP(Module):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 hidden_act,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 quant_mode=QuantMode(0),
                 instance_id: int = 0):
        super().__init__()
        if hidden_act not in ACT2FN:
            raise ValueError(
                'unsupported activation function: {}'.format(hidden_act))
        fc_output_size = 2 * ffn_hidden_size if hidden_act == 'swiglu' else ffn_hidden_size
        self.use_fp8_qdq = quant_mode.has_fp8_qdq()

        if self.use_fp8_qdq:
            self.fc = FP8Linear(hidden_size,
                                fc_output_size,
                                bias=bias,
                                dtype=dtype,
                                tp_group=tp_group,
                                tp_size=tp_size,
                                gather_output=False)
            self.proj = FP8RowLinear(ffn_hidden_size,
                                     hidden_size,
                                     bias=bias,
                                     dtype=dtype,
                                     tp_group=tp_group,
                                     tp_size=tp_size,
                                     instance_id=instance_id)
        else:
            self.fc = ColumnLinear(hidden_size,
                                   fc_output_size,
                                   bias=bias,
                                   dtype=dtype,
                                   tp_group=tp_group,
                                   tp_size=tp_size,
                                   gather_output=False)
            self.proj = RowLinear(ffn_hidden_size,
                                  hidden_size,
                                  bias=bias,
                                  dtype=dtype,
                                  tp_group=tp_group,
                                  tp_size=tp_size,
                                  instance_id=instance_id)

        self.hidden_act = hidden_act
        self.dtype = dtype

    def forward(self, hidden_states, workspace=None, lora_param=None):
        mlp_h_to_4h_param = None
        if lora_param is not None:
            mlp_h_to_4h_param = lora_param.get_runtime_params(0, "mlp_h_to_4h")

        mlp_4h_to_h_param = None
        if lora_param is not None:
            mlp_4h_to_h_param = lora_param.get_runtime_params(0, "mlp_4h_to_h")

        inter = self.fc(hidden_states, mlp_h_to_4h_param)
        inter = ACT2FN[self.hidden_act](inter)
        output = self.proj(inter,
                           workspace,
                           lora_runtime_param=mlp_4h_to_h_param)
        return output


class GatedMLP(MLP):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 hidden_act,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 quant_mode=QuantMode(0),
                 instance_id: int = 0):
        self.use_fp8_qdq = quant_mode.has_fp8_qdq()
        super().__init__(hidden_size,
                         ffn_hidden_size,
                         hidden_act,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         quant_mode=quant_mode,
                         instance_id=instance_id)

        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.hidden_act = hidden_act
        self.bias = bias
        self.dtype = dtype
        self.tp_group = tp_group
        self.tp_size = tp_size
        self.quant_mode = quant_mode
        self.instance_id = instance_id

        if self.use_fp8_qdq:
            self.gate = FP8Linear(hidden_size,
                                  ffn_hidden_size,
                                  bias=bias,
                                  dtype=dtype,
                                  tp_group=tp_group,
                                  tp_size=tp_size,
                                  gather_output=False)
        else:
            self.gate = ColumnLinear(hidden_size,
                                     ffn_hidden_size,
                                     bias=bias,
                                     dtype=dtype,
                                     tp_group=tp_group,
                                     tp_size=tp_size,
                                     gather_output=False)

    def forward(self, hidden_states, workspace=None, lora_param=None):

        mlp_h_to_4h_param = None
        if lora_param is not None:
            mlp_h_to_4h_param = lora_param.get_runtime_params(0, "mlp_h_to_4h")

        mlp_gate_param = None
        if lora_param is not None:
            mlp_gate_param = lora_param.get_runtime_params(0, "mlp_gate")

        mlp_4h_to_h_param = None
        if lora_param is not None:
            mlp_4h_to_h_param = lora_param.get_runtime_params(0, "mlp_4h_to_h")

        inter = self.fc(hidden_states, mlp_h_to_4h_param)
        inter = ACT2FN[self.hidden_act](inter)
        gate = self.gate(hidden_states, mlp_gate_param)
        intermediate = inter * gate
        output = self.proj(intermediate,
                           workspace,
                           lora_runtime_param=mlp_4h_to_h_param)
        return output


class FusedGatedMLP(GatedMLP):

    def __init__(self,
                 hidden_size,
                 ffn_hidden_size,
                 hidden_act,
                 bias=True,
                 dtype=None,
                 tp_group=None,
                 tp_size=1,
                 quant_mode=QuantMode(0),
                 instance_id: int = 0):
        super().__init__(hidden_size,
                         ffn_hidden_size,
                         hidden_act,
                         bias=bias,
                         dtype=dtype,
                         tp_group=tp_group,
                         tp_size=tp_size,
                         quant_mode=quant_mode,
                         instance_id=instance_id)

    def forward(self, hidden_states, workspace=None, lora_param=None):
        # Combine the following pattern
        #
        #   SiLU(FC(x)) + Gate(x)
        #
        # into:
        #
        #   SwiGLU(FusedFC(x))
        #
        # Upside is we don't need to modify 4 different weight loading paths just to concat weights

        _np_dtype = trt_dtype_to_np(self.dtype)
        concat_weight = np.concatenate(
            [self.gate.weight.raw_value, self.fc.weight.raw_value],
            axis=0).astype(_np_dtype)
        if self.bias:
            concat_bias = np.concatenate(
                [self.gate.bias.raw_value, self.fc.bias.raw_value],
                axis=0).astype(_np_dtype)

        if self.use_fp8_qdq:
            gate_weights_scaling_factor = self.gate.weights_scaling_factor.raw_value
            fc_weights_scaling_factor = self.fc.weights_scaling_factor.raw_value
            fc_activation_scaling_factor = self.fc.activation_scaling_factor.raw_value
            gate_activation_scaling_factor = self.gate.activation_scaling_factor.raw_value
            assert fc_activation_scaling_factor == gate_activation_scaling_factor, "Activation scales should be identical"

        # Remove dangling TRT-LLM parameter references after the graph rewrite.
        for param, _ in list(self.gate.named_parameters()):
            self.gate._parameters.pop(param)
        self.gate = None

        if self.use_fp8_qdq:
            self.fc = FP8Linear(self.hidden_size,
                                self.ffn_hidden_size * 2,
                                bias=self.bias,
                                dtype=self.dtype,
                                tp_group=self.tp_group,
                                tp_size=self.tp_size,
                                gather_output=False)
        else:
            self.fc = ColumnLinear(self.hidden_size,
                                   self.ffn_hidden_size * 2,
                                   bias=self.bias,
                                   dtype=self.dtype,
                                   tp_group=self.tp_group,
                                   tp_size=self.tp_size,
                                   gather_output=False)

        self.fc.weight.value = concat_weight

        if self.use_fp8_qdq:
            self.fc.activation_scaling_factor.value = fc_activation_scaling_factor
            # TODO: need to align with quantization toolkit; preferably put a constraint to equalize
            # fc/gate weight scaling factor to allow horizontal fusion without accuracy loss
            self.fc.weights_scaling_factor.value = max(
                gate_weights_scaling_factor, fc_weights_scaling_factor)

        if self.bias:
            self.fc.bias.value = concat_bias
        inter = self.fc(hidden_states)
        if self.hidden_act == 'silu':
            inter = ACT2FN['swiglu'](inter)
        else:
            raise NotImplementedError(
                f"Activation {self.hidden_act} not yet implemented for FusedGatedMLP"
            )
        output = self.proj(inter, workspace)
        return output

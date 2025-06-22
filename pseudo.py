from mx import finalize_mx_specs, mx_mapping
from mx.mx_ops import quantize_mx_op
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


class Mx:
    def __init__(
        self,
        w_format="fp4_e2m1",
        a_format="fp4_e2m1",
        block_size=32,
        bfloat=0,  # 0 means don't use
        fp=16,
        custom_cuda=True,
        quantize_backprop=False,
        round="even",
    ):
        mx_specs = {
            "w_elem_format": w_format,
            "a_elem_format": a_format,
            "block_size": block_size,
            "bfloat": bfloat,
            "fp": fp,
            "custom_cuda": custom_cuda,
            "quantize_backprop": quantize_backprop,
            "round": round,
        }
        self.mx_specs = finalize_mx_specs(mx_specs)
        mx_mapping.inject_pyt_ops(self.mx_specs)

    @torch.no_grad()
    def pseudo_quantize(self, model):
        element_format = self.mx_specs["w_elem_format"]
        layers = model.model.layers
        for layer in layers:
            if isinstance(layer, (nn.LayerNorm, LlamaRMSNorm, Qwen2RMSNorm)):
                layer.weight = nn.Parameter(
                    quantize_mx_op(
                        layer.weight,
                        self.mx_specs,
                        element_format,
                        self.mx_specs["block_size"],
                        1,
                    )
                )
        return model

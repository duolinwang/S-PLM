# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union
import torch
import torch.nn as nn

import esm_adapterH
from esm_adapterH.modules import ContactPredictionHead, ESM1bLayerNorm, RobertaLMHead, TransformerLayer, TransformerAdapterLayer
from esm_adapterH.adapter import ResMLP

from collections import defaultdict


class ESM2(nn.Module):
    def __init__(
        self,
        num_layers: int = 33,
        embed_dim: int = 1280,
        attention_heads: int = 20,
        alphabet: Union[esm_adapterH.data.Alphabet, str] = "ESM-1b",
        token_dropout: bool = True,
        #num_end_adapter_layers=None,
        adapter_args = None,
        prefix_module: nn.Module = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        if not isinstance(alphabet, esm_adapterH.data.Alphabet):
            alphabet = esm_adapterH.data.Alphabet.from_architecture(alphabet)
        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.padding_idx = alphabet.padding_idx
        self.mask_idx = alphabet.mask_idx
        self.cls_idx = alphabet.cls_idx
        self.eos_idx = alphabet.eos_idx
        self.prepend_bos = alphabet.prepend_bos
        self.append_eos = alphabet.append_eos
        self.token_dropout = token_dropout
        self.adapter_args=adapter_args
        self.prefix_module = prefix_module
        if adapter_args is None:
           self._init_submodules()
        else:
           self._init_submodules_adapter()
        
        self.embeding_coder = None
    
    def _init_submodules(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )

    def _init_submodules_adapter(self):
        self.embed_scale = 1
        self.embed_tokens = nn.Embedding(
            self.alphabet_size,
            self.embed_dim,
            padding_idx=self.padding_idx,
        )


        # Idea from "Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation"
        # https://arxiv.org/abs/2304.12620
        # https://github.com/WuJunde/Medical-SAM-Adapter/tree/main
        
        
        # self.adapter_args.num_end_adapter_layers=[3,2,3]
        # =>
        # {
        # 3: ['adapter_0', 'adapter_2']
        # 4: ['adapter_0', 'adapter_1', 'adapter_2']
        # 5: ['adapter_0', 'adapter_1', 'adapter_2']
        # }
        adapter_layer_idx_dict = {}
        if not isinstance(self.adapter_args.num_end_adapter_layers, list):
           self.adapter_args.num_end_adapter_layers = [self.adapter_args.num_end_adapter_layers]
        
        layer_adapter_dict = defaultdict(list)
        for adapter_idx, value in enumerate(self.adapter_args.num_end_adapter_layers):
          adapter_name = f"adapter_{adapter_idx}"
          adapter_layer_idx_dict[adapter_name] = []
          start_layer_idx = self.num_layers - value
          if start_layer_idx < 0:
            start_layer_idx = 0

          for layer_idx in range(start_layer_idx, self.num_layers):
            layer_adapter_dict[layer_idx].append(adapter_name)
        
        
        self.adapter_layer = {}
        for idx in range(self.num_layers):
          self.adapter_layer[idx] = None
          
        for layer_idx, adapter_name_list in layer_adapter_dict.items():
          self.adapter_layer[layer_idx] = []
          module_dict = nn.ModuleDict({})
          for adapter_name in adapter_name_list:
            module_dict[adapter_name] = nn.ModuleList([])
            for _ in range(2):
              module_dict[adapter_name].append(
                ResMLP(
                bottleneck_size=self.embed_dim // 2,
                module_type=self.adapter_args.module_type, #"MLP1",
                dropout=0,
                emb_dimension=self.embed_dim,
                nonlinearity='relu',
                layer_norm=True,
                residual=True,
                )
              )
          self.adapter_layer[layer_idx] = module_dict

        
        self.layers = nn.ModuleList(
            [
                TransformerAdapterLayer(
                    self.embed_dim,
                    4 * self.embed_dim,
                    self.attention_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                    adapter_layer_dict=self.adapter_layer[idx],
                )
                for idx in range(self.num_layers)
            ]
        )

        self.contact_head = ContactPredictionHead(
            self.num_layers * self.attention_heads,
            self.prepend_bos,
            self.append_eos,
            eos_idx=self.eos_idx,
        )
        self.emb_layer_norm_after = ESM1bLayerNorm(self.embed_dim)

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=self.alphabet_size,
            weight=self.embed_tokens.weight,
        )


    def forward(self, tokens, task_ids = None, repr_layers=[], need_head_weights=False, return_contacts=False):
        if return_contacts:
            need_head_weights = True

        assert tokens.ndim == 2

        is_previoues_prompted = False
        # if (self.prefix_module is not None and 
        #     0 in self.prefix_module.prompt_layer_indices):
        #     # prefill_lengths = tokens.shape[1] + prompt_len
        #     # just give an unique token id to prompt tokens
        #     prompt_token_idx = len(self.alphabet.all_toks)+1
        #     prompt_token = torch.full(
        #         (tokens.shape[0], self.prefix_module.prompt_len),
        #         prompt_token_idx).to(tokens.device)

        #     # Concatenate the tensors and reshape. Shape [B, prompt_length+T]
        #     concated_tokens = torch.cat(
        #         (prompt_token, tokens), dim=1)  
            
        #     # Shape = [B, (prompt_len+T)]
        #     padding_mask = concated_tokens.eq(
        #         self.padding_idx)
            
        #     x = self.embed_scale * self.embed_tokens(tokens)
        #     x = self.prefix_module(x, 0)

        #     tokens = concated_tokens
        #     is_prompted = True
        # else:
        #     padding_mask = tokens.eq(self.padding_idx)  # B, T
        #     x = self.embed_scale * self.embed_tokens(tokens)
        

        padding_mask = tokens.eq(self.padding_idx)  # B, T
        x = self.embed_scale * self.embed_tokens(tokens)

        if self.token_dropout:
            x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
            # x: B x T x C
            mask_ratio_train = 0.15 * 0.8
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (
                tokens == self.mask_idx).sum(-1).to(x.dtype) / src_lengths
            x = x * (1 - mask_ratio_train) / \
                (1 - mask_ratio_observed)[:, None, None]

        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        repr_layers = set(repr_layers)
        hidden_representations = {}
        if 0 in repr_layers:
            hidden_representations[0] = x

        if need_head_weights:
            attn_weights = []

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        if padding_mask is not None and self.prefix_module is not None:
            prefix_padding_mask = torch.zeros(padding_mask.shape[0], 
                                              self.prefix_module.prompt_len, 
                                              dtype=torch.bool, 
                                              device=padding_mask.device)
            padding_mask = torch.cat((prefix_padding_mask, padding_mask), dim=1)
        #"""
        
        for layer_idx, layer in enumerate(self.layers):
            
            
            if (self.prefix_module is not None and 
                layer_idx in self.prefix_module.prompt_layer_indices):
                # (prompt_len+T, B, E) => (B, prompt_len+T, E)
                x = x.transpose(0, 1)
                
                if is_previoues_prompted:
                  # remove the prompt_tokens
                  # (B, prompt_len+T, E) => (B, T, E)
                  x = x[:, self.prefix_module.prompt_len:, :]
                
                x = self.prefix_module(x, layer_idx, task_ids)
                
                # (B, prompt_len+T, E) => (prompt_len+T, B, E)
                x = x.transpose(0, 1)

                is_previoues_prompted = True

            x, attn = layer(
                x,
                self_attn_padding_mask=padding_mask,
                need_head_weights=need_head_weights,
            )
            if (layer_idx + 1) in repr_layers:
                hidden_representations[layer_idx + 1] = x.transpose(0, 1)
                
            if need_head_weights:
                # (H, B, T, T) => (B, H, T, T)
                attn_weights.append(attn.transpose(1, 0))

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)
        
        if self.prefix_module is not None:
            # remove the prompt_tokens
            x = x[:, self.prefix_module.prompt_len:, :]  # => [B,T,E]

        # last hidden representation should have layer norm applied
        if (layer_idx + 1) in repr_layers:
            hidden_representations[layer_idx + 1] = x
        x = self.lm_head(x)

        result = {"logits": x, "representations": hidden_representations}
        if need_head_weights:
            # attentions: B x L x H x T x T
            attentions = torch.stack(attn_weights, 1)
            if padding_mask is not None:
                attention_mask = 1 - padding_mask.type_as(attentions)
                attention_mask = attention_mask.unsqueeze(
                    1) * attention_mask.unsqueeze(2)
                attentions = attentions * attention_mask[:, None, None, :, :]
            result["attentions"] = attentions
            if return_contacts:
                contacts = self.contact_head(tokens, attentions)
                result["contacts"] = contacts

        return result

    def predict_contacts(self, tokens):
        return self(tokens, return_contacts=True)["contacts"]

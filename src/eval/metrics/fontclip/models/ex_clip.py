from collections import OrderedDict
from typing import Union, Tuple, List, Iterator
import torch
from torch import nn
from torch.nn import Parameter
import numpy as np

from models.ex_clip_multiheadattention import ExMultiheadAttention
from models.oft import OFTConfig
from models.lora import LoRAConfig

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ExResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        use_attention_hook: bool = False,
        use_oft: bool = False,
        oft_config: OFTConfig = None,
        use_lora: bool = False,
        lora_config: LoRAConfig = None,
    ):
        super().__init__()

        self.attn = ExMultiheadAttention(
            d_model,
            n_head,
            use_oft=use_oft,
            oft_config=oft_config,
            use_lora=use_lora,
            lora_config=lora_config,
        )
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.use_attention_hook = use_attention_hook
        self.attn_probs = None
        self.attn_grad = None

    def set_attn_probs(self, attn_probs):
        self.attn_probs = attn_probs

    def set_attn_grad(self, attn_grad):
        self.attn_grad = attn_grad

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        if self.use_attention_hook:
            return self.attn(
                x,
                x,
                x,
                need_weights=False,
                attn_mask=self.attn_mask,
                attention_probs_forward_hook=self.set_attn_probs,
                attention_probs_backward_hook=self.set_attn_grad,
            )[0]
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ExTransformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        return_each_resblock_output: bool = False,
        use_attention_hook: bool = False,
        use_oft: bool = False,
        oft_config: OFTConfig = None,
        use_lora: bool = False,
        lora_config: LoRAConfig = None,
        use_pt: bool = False,
        precontext_length: int = 0,
        pt_applied_layers: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        precontext_dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[
                ExResidualAttentionBlock(
                    width,
                    heads,
                    attn_mask,
                    use_attention_hook=use_attention_hook,
                    use_oft=use_oft,
                    oft_config=oft_config,
                    use_lora=use_lora,
                    lora_config=lora_config,
                )
                for _ in range(layers)
            ]
        )
        self.return_each_resblock_output = return_each_resblock_output
        self.use_pt = use_pt
        self.precontext_length = precontext_length
        self.pt_applied_layers = pt_applied_layers
        self.precontext_dropout_rate = precontext_dropout_rate

    def forward(self, x: torch.Tensor, precontext_tokens: List[torch.Tensor] = None):
        _, B, _ = x.shape
        if self.return_each_resblock_output:
            outputs = []
            for resblock in self.resblocks:
                x = resblock(x)
                outputs.append(x)
            return outputs
        elif not self.use_pt:
            return self.resblocks(x)
        else:
            applied_precontext_count = 0
            for i, resblock in enumerate(self.resblocks):
                # We already inject precontext_visions[0] in the first layer.
                x = resblock(x)
                if i + 1 in self.pt_applied_layers:
                    # LND -> NLD
                    x = x.permute(1, 0, 2)
                    # replace x[:, 1:self.precontext_length+1, :] with precontext_visions[applied_precontext_vision_count+1]
                    x = torch.cat(
                        [
                            x[:, :1, :],
                            self.precontext_dropout(
                                precontext_tokens[applied_precontext_count + 1].expand(
                                    B, -1, -1
                                )
                            ),
                            x[:, 1 + self.precontext_length :, :],
                        ],
                        dim=1,
                    )
                    # NLD -> LND
                    x = x.permute(1, 0, 2)
                    applied_precontext_count += 1
            return x


class ExVisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        return_each_resblock_output: bool = False,
        use_attention_hook: bool = False,
        use_oft: bool = False,
        oft_config: OFTConfig = None,
        use_lora: bool = False,
        lora_config: LoRAConfig = None,
        use_pt: bool = False,
        precontext_length: int = 0,
        pt_applied_layers: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        precontext_dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.return_each_resblock_output = return_each_resblock_output

        # prompt tuning settings
        self.use_pt = use_pt
        if self.use_pt:
            self.precontext_length = precontext_length
            self.pt_applied_layers = pt_applied_layers
            self.precontext_dropout_rate = precontext_dropout_rate
            self.precontext_dropout = nn.Dropout(p=self.precontext_dropout_rate)
            self.precontext_tokens = []
            print(f"precontext_length: {self.precontext_length}")
            print(f"pt_applied_layers: {self.pt_applied_layers}")
            for layer_i in self.pt_applied_layers:
                param = nn.Parameter(
                    torch.randn(
                        1,
                        self.precontext_length,
                        width,
                        device=device,
                    )
                )
                # initalization is crucial for stable training
                nn.init.normal_(param, std=0.05)
                self.precontext_tokens.append(param)
                self.register_parameter(f"precontext_token{layer_i}", param)
        else:
            self.precontext_length = 0
            self.pt_applied_layers = []
            self.precontext_dropout_rate = 0
            self.precontext_dropout = None
            self.precontext_tokens = None

        self.transformer = ExTransformer(
            width,
            layers,
            heads,
            return_each_resblock_output=self.return_each_resblock_output,
            use_attention_hook=use_attention_hook,
            use_oft=use_oft,
            oft_config=oft_config,
            use_lora=use_lora,
            lora_config=lora_config,
            use_pt=self.use_pt,
            precontext_length=self.precontext_length,
            pt_applied_layers=self.pt_applied_layers,
            precontext_dropout_rate=self.precontext_dropout_rate,
        )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        # x.shape = [*, 3, input_resolution, input_resolution]
        B, _, _, _ = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid] default: [*, 768, 7, 7]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        # x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)
        if self.use_pt:
            x = torch.cat(
                [
                    x[:, :1, :],
                    self.precontext_dropout(
                        self.precontext_tokens[0].expand(B, -1, -1)
                    ),
                    x[:, 1:, :],
                ],
                dim=1,
            ).to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND [grid ** 2 + 1, *, width]
        if self.return_each_resblock_output:
            outputs = self.transformer(x, self.precontext_tokens)
            x = outputs.pop()
        else:
            x = self.transformer(x, self.precontext_tokens)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        if self.return_each_resblock_output:
            return x, outputs
        return x


class ExCLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        return_each_resblock_output: bool = False,
        use_attention_hook: bool = False,
        use_oft_vision: bool = False,
        use_oft_text: bool = False,
        oft_config_vision: OFTConfig = None,
        oft_config_text: OFTConfig = None,
        use_lora_vision: bool = False,
        use_lora_text: bool = False,
        lora_config_vision: LoRAConfig = None,
        lora_config_text: LoRAConfig = None,
        use_coop_vision: bool = False,
        use_coop_text: bool = False,
        precontext_length_vision: int = 10,
        precontext_length_text: int = 16,
        precontext_dropout_rate: int = 0.1,
        pt_applied_layers: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ):
        super().__init__()

        self.use_oft_vision = use_oft_vision
        self.use_oft_text = use_oft_text
        self.use_lora_vision = use_lora_vision
        self.use_lora_text = use_lora_text
        self.use_attention_hook = use_attention_hook
        self.return_each_resblock_output = return_each_resblock_output
        self.context_length = context_length
        self.use_coop_vision = use_coop_vision
        self.use_coop_text = use_coop_text

        vision_heads = vision_width // 64
        # print('VisionTransformer')
        self.visual = ExVisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            return_each_resblock_output=self.return_each_resblock_output,
            use_attention_hook=self.use_attention_hook,
            use_oft=use_oft_vision,
            oft_config=oft_config_vision,
            use_lora=use_lora_vision,
            lora_config=lora_config_vision,
            use_pt=use_coop_vision,
            precontext_length=precontext_length_vision,
            pt_applied_layers=pt_applied_layers,
            precontext_dropout_rate=precontext_dropout_rate,
        )

        self.transformer = ExTransformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            use_attention_hook=self.use_attention_hook,
            use_oft=use_oft_text,
            oft_config=oft_config_text,
            use_lora=use_lora_text,
            lora_config=lora_config_text,
        )
        if self.use_coop_text:
            self.precontext_length_text = precontext_length_text
            self.precontext_text = nn.Parameter(
                torch.randn(
                    1,
                    precontext_length_text,
                    self.transformer.width,
                    device=device,
                    dtype=self.dtype,
                )
            )
            nn.init.normal_(self.precontext_text, std=0.02)
            self.register_parameter("precontext_text", self.precontext_text)

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width)
        )
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def coop_parameters(self):
        for coop_type in ["text", "vision"]:
            if coop_type == "text" and self.use_coop_text:
                yield self.precontext_text
            elif coop_type == "vision" and self.use_coop_vision:
                for name, param in self.visual.named_parameters(recurse=True):
                    if name.startswith("precontext_token"):
                        yield param

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            if (name not in ["precontext_text"]) and (
                not name.startswith("visual.precontext_token")
            ):
                yield param
    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        if isinstance(self.visual, ExVisionTransformer):
            for block in self.visual.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
                # if enabling the following two lines, "RuntimeWarning: invalid value encountered in true_divide" occurs
                # nn.init.normal_(block.ln_1.weight, std=proj_std)
                # nn.init.normal_(block.ln_2.weight, std=proj_std)

        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
            # if enabling the following two lines, "RuntimeWarning: invalid value encountered in true_divide" occurs
            # nn.init.normal_(block.ln_1.weight, std=proj_std)
            # nn.init.normal_(block.ln_2.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        if self.use_coop_text:
            context = self.precontext_text
            K, N1, C1 = x.shape
            B, N2, C2 = context.shape
            assert C1 == C2
            eos_indx = text.argmax(dim=-1) + N2
            eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)
            x = x.reshape(1, K, N1, C1).expand(B, K, N1, C1)
            context = (
                context.expand(B, N2, C1).reshape(B, 1, N2, C1).expand(B, K, N2, C1)
            )
            x = (
                torch.cat([x[:, :, 0:1], context, x[:, :, 1:]], dim=2)
                .reshape(B * K, N1 + N2, C1)
                .type(self.dtype)
            )
            # TODO: implement the other two types of coop, post and mid.
            # elif self.token_type == "post":
            #     K, N1, C1 = x.shape
            #     B, N2, C2 = context.shape
            #     assert C1 == C2
            #     eos_indx = text.argmax(dim=-1)
            #     eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)
            #     x = x.reshape(1, K, N1, C1).expand(B, K, N1, C1)
            #     context = (
            #         context.expand(B, N2, C1).reshape(B, 1, N2, C1).expand(B, K, N2, C1)
            #     )
            #     x = (
            #         torch.cat([x, context], dim=2)
            #         .reshape(B * K, N1 + N2, C1)
            #         .type(self.dtype)
            #     )

        else:
            # If not self.text_coop, follow the original CLIP code.
            eos_indx = text.argmax(dim=-1)

        # refer to https://github.com/wenwenyu/TCM/blob/cfa4756f4082d7d00e76161fb81dfa2c079d181c/ocrclip/ocrclip/models.py#L971C18-L971C18

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number
        # in each sequence)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        return x

    def forward(self, image, text):
        if self.return_each_resblock_output:
            image_features, outputs = self.encode_image(image)
        else:
            image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        # logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

import hashlib
import os
from PIL import Image
import torch
from torch import nn
from tqdm import tqdm
from typing import Union, List
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import urllib
import warnings

from models.ex_clip_multiheadattention import ExMultiheadAttention
from models.ex_clip import ExCLIP
from utils.transform_image import my_transform
from models.oft import OFTConfig
from models.lora import LoRAConfig

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B/32"
# model_name = "ViT-L/14"

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

__all__ = ["available_models", "load", "tokenize"]

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose(
        [
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention) or isinstance(l, ExMultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
                # OFT
                *[f"oft_{s}" for s in ["q", "v", "k", "out",]],
                # LoRA
                *[f"{s}_lora_proj_weight_a" for s in ["q", "v", "k", "out",]],
                *[f"{s}_lora_proj_weight_b" for s in ["q", "v", "k", "out",]],
                *[f"{s}_lora_proj_bias" for s in ["q", "v", "k", "out",]],
            ]:
                try:
                    tensor = getattr(l, attr)
                    if tensor is not None:
                        tensor.data = tensor.data.half()
                except AttributeError as e:
                    pass

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(
    state_dict: dict,
    vision_layers: int = None,
    transformer_layers: int = None,
    load_state_dict: bool = True,
    use_attention_hook: bool = False,
    use_oft_vision: bool = False,
    use_oft_text: bool = False,
    oft_config_vision: OFTConfig=None,
    oft_config_text: OFTConfig=None,
    use_lora_vision: bool = False,
    use_lora_text: bool = False,
    lora_config_vision: LoRAConfig=None,
    lora_config_text: LoRAConfig=None,
    use_coop_vision: bool = False,
    use_coop_text: bool = False,
    precontext_length_text: int = 16,
    precontext_length_vision: int = 10,
    precontext_dropout_rate: int = 0.1,
    pt_applied_layers: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        if vision_layers is None:
            vision_layers = len(
                [
                    k
                    for k in state_dict.keys()
                    if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
                ]
            )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        if vision_layers is None:
            vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width**2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    if transformer_layers is None:
        transformer_layers = len(
            set(
                k.split(".")[2]
                for k in state_dict
                if k.startswith("transformer.resblocks")
            )
        )

    model = ExCLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        use_attention_hook=use_attention_hook,
        use_oft_vision=use_oft_vision,
        use_oft_text=use_oft_text,
        oft_config_vision=oft_config_vision,
        oft_config_text=oft_config_text,
        use_lora_vision=use_lora_vision,
        use_lora_text=use_lora_text,
        lora_config_vision=lora_config_vision,
        lora_config_text=lora_config_text,
        use_coop_vision=use_coop_vision,
        use_coop_text=use_coop_text,
        precontext_length_vision=precontext_length_vision,
        precontext_length_text=precontext_length_text,
        precontext_dropout_rate=precontext_dropout_rate,
        pt_applied_layers=pt_applied_layers,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    for key in list(state_dict.keys()):
        if key not in model.state_dict():
            del state_dict[key]

    convert_weights(model)
    if load_state_dict:
        model.load_state_dict(state_dict, strict=False)
    else:
        # I think the parameters are already initialized in __init__(), so we don't need to initialize them again, but just in case
        model.initialize_parameters()
    convert_weights(model)
    return model.eval()


def load(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit: bool = False,
    download_root: str = None,
    vision_layers: int = None,
    transformer_layers: int = None,
    load_state_dict: bool = True,
    use_attention_hook: bool = False,
    use_oft_vision: bool = False,
    use_oft_text: bool = False,
    oft_config_vision: OFTConfig=None,
    oft_config_text: OFTConfig=None,
    use_lora_vision: bool = False,
    use_lora_text: bool = False,
    lora_config_vision: LoRAConfig=None,
    lora_config_text: LoRAConfig=None,
    use_coop_vision: bool = False,
    use_coop_text: bool = False,
    precontext_length_text: int = 16,
    precontext_length_vision: int = 10,
    precontext_dropout_rate: int = 0.1,
    pt_applied_layers: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(
            _MODELS[name], download_root or os.path.expanduser("~/.cache/clip")
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with open(model_path, "rb") as opened_file:
        try:
            # loading JIT archive
            print("loading JIT archive", model_path)
            model = torch.jit.load(
                opened_file, map_location=device if jit else "cpu"
            ).eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(
                    f"File {model_path} is not a JIT archive. Loading as a state dict instead"
                )
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(
            state_dict or model.state_dict(),
            vision_layers=vision_layers,
            transformer_layers=transformer_layers,
            load_state_dict=load_state_dict,
            use_attention_hook=use_attention_hook,
            use_oft_vision=use_oft_vision,
            use_oft_text=use_oft_text,
            oft_config_vision=oft_config_vision,
            oft_config_text=oft_config_text,
            use_lora_vision=use_lora_vision,
            use_lora_text=use_lora_text,
            lora_config_vision=lora_config_vision,
            lora_config_text=lora_config_text,
            use_coop_vision=use_coop_vision,
            use_coop_text=use_coop_text,
            precontext_length_vision=precontext_length_vision,
            precontext_length_text=precontext_length_text,
            precontext_dropout_rate=precontext_dropout_rate,
            pt_applied_layers=pt_applied_layers,
        ).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
    )
    device_node = [
        n
        for n in device_holder.graph.findAllNodes("prim::Constant")
        if "Device" in repr(n)
    ][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith(
                    "cuda"
                ):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[]
        )
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [
                        1,
                        2,
                    ]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())


def load_model(
    checkpoint_path=None,
    requires_grad=False,
    device=device,
    model_name=model_name,
    vision_layers=None,
    transformer_layers=None,
    load_state_dict=True,
    use_attention_hook=False,
    use_lora_text=False,
    use_lora_vision=False,
    lora_config_vision: LoRAConfig=None,
    lora_config_text: LoRAConfig=None,
    use_oft_vision=False,
    use_oft_text=False,
    oft_config_vision: OFTConfig=None,
    oft_config_text: OFTConfig=None,
    use_coop_vision: bool = False,
    use_coop_text: bool = False,
    precontext_length_vision: int = 10,
    precontext_length_text: int = 16,
    precontext_dropout_rate: int = 0.1,
    pt_applied_layers: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
):
    if checkpoint_path is not None:
        assert load_state_dict
    model, _ = load(
        model_name,
        device=device,
        jit=False,
        vision_layers=vision_layers,
        transformer_layers=transformer_layers,
        load_state_dict=load_state_dict,
        use_attention_hook=use_attention_hook,
        use_oft_vision=use_oft_vision,
        use_oft_text=use_oft_text,
        oft_config_vision=oft_config_vision,
        oft_config_text=oft_config_text,
        use_lora_text=use_lora_text,
        use_lora_vision=use_lora_vision,
        lora_config_vision=lora_config_vision,
        lora_config_text=lora_config_text,
        use_coop_vision=use_coop_vision,
        use_coop_text=use_coop_text,
        precontext_length_vision=precontext_length_vision,
        precontext_length_text=precontext_length_text,
        precontext_dropout_rate=precontext_dropout_rate,
        pt_applied_layers=pt_applied_layers,
    )
    if checkpoint_path is not None:
        print("init_model: loading checkpoint", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    if not requires_grad:
        for param in model.parameters():
            param.requires_grad = False
    return model


model, preprocess = load(model_name, device=device, jit=False)
my_preprocess = my_transform(model.visual.input_resolution)
preprocess_for_single_character = my_transform(
    model.visual.input_resolution,
    lower_bound_of_scale=0.85,
    upper_bound_of_scale=1.0,
)
preprocess_for_aug = my_transform(
    model.visual.input_resolution, do_aug=True, do_normalize=False
)
preprocess_for_normalize = my_transform(
    model.visual.input_resolution, do_aug=False, do_normalize=True
)
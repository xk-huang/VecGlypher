from models.init_model import device, load_model, my_preprocess, preprocess
from models.lora import LoRAConfig
from utils.tokenizer import tokenize


def main():
    # download ckpt from: https://drive.google.com/uc?id=1Tym7rAIuaGr6Gv-gZRSJmPstQjOWPgl1
    checkpoint_path = "/home/vecglypher/mnt/workspace/hf_downloads/fontclip_weight/ViT-B_32_bce_lora_t-qkvo_256-1024.0_91011_batch64_aug200_lbound_of_scale0.35_max_attr_num_3_random_p_num_30000_geta0.2_use_negative_lr0.0002-0.1_image_file_dir.pt"

    lora_config_text = LoRAConfig(
        r=256,
        alpha=1024.0,
        bias=False,
        learnable_alpha=False,
        apply_q=True,
        apply_k=True,
        apply_v=True,
        apply_out=True,
    )
    model = load_model(
        checkpoint_path,
        model_name="ViT-B/32",
        use_oft_vision=False,
        use_oft_text=False,
        oft_config_vision=None,
        oft_config_text=None,
        use_lora_text=True,
        use_lora_vision=False,
        lora_config_vision=None,
        lora_config_text=lora_config_text,
        use_coop_text=False,
        use_coop_vision=False,
        precontext_length_vision=10,
        precontext_length_text=77,
        precontext_dropout_rate=0,
        pt_applied_layers=None,
    )


if __name__ == "__main__":
    main()

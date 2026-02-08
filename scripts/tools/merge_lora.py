"""
base_model_path_list=(
    workspace/hf_downloads/google/gemma-3-4b-it
    workspace/hf_downloads/google/gemma-3-4b-it
    workspace/hf_downloads/google/gemma-3-27b-it
    workspace/hf_downloads/google/gemma-3-27b-it
)

lora_model_path_list=(
    workspace/svg_glyph_llm/saves/251117-ablate-lora/4b-rel_coord
    workspace/svg_glyph_llm/saves/251117-ablate-lora/4b-abs_coord
    workspace/svg_glyph_llm/saves/251117-ablate-lora/27b-rel_coord
    workspace/svg_glyph_llm/saves/251117-ablate-lora/27b-abs_coord
)

output_model_path_list=(
    saves/251117-ablate-lora-merged/4b-rel_coord
    saves/251117-ablate-lora-merged/4b-abs_coord
    saves/251117-ablate-lora-merged/27b-rel_coord
    saves/251117-ablate-lora-merged/27b-abs_coord
)

mnt_storage_dir=/home/vecglypher/mnt/
base_storage_dir=workspace/svg_glyph_llm/
template=gemma3

length=${#base_model_path_list[@]}

for ((idx=1; idx<=length; idx++)); do
    echo "idx: ${idx}"

    base_model_path=${base_model_path_list[$idx]}
    local_base_model_path=saves/merge_lora_temp/$(basename $base_model_path)
    echo "base_model_path: ${base_model_path}"

    lora_model_path=${lora_model_path_list[$idx]}
    local_lora_model_path=saves/merge_lora_temp/$(basename $lora_model_path)
    echo "lora_model_path: ${lora_model_path}"
    if [[ ! -d ${local_base_model_path} ]]; then
        echo "base downloading: $base_model_path"
        storage_cli --prod-use-cython-client getr $base_model_path ${local_base_model_path} --threads 20 --jobs 10 > /dev/null 2>&1
    else
        echo "base exists: $base_model_path"
    fi
    if [[ ! -d ${local_lora_model_path} ]]; then
        echo "lora downloading: $lora_model_path"
        storage_cli --prod-use-cython-client getr $lora_model_path ${local_lora_model_path} --threads 20 --jobs 10 > /dev/null 2>&1
    else
        echo "lora exists: $lora_model_path"
    fi

    output_model_path=${output_model_path_list[$idx]}
    echo "output_model_path: ${output_model_path}"

    python scripts/tools/merge_lora.py \
        "${local_base_model_path}" \
        "${local_lora_model_path}" \
        "${template}" \
        "${output_model_path}"
done

storage_cli --prod-use-cython-client putr saves/251117-ablate-lora-merged  workspace/svg_glyph_llm/saves/251117-ablate-lora-merged --threads 20 --jobs 10
"""

import subprocess
import tempfile

import click
import yaml

# https://github.com/hiyouga/LLaMA-Factory/blob/9779b1f361156bc361aea32382d94f68dd92d028/examples/merge_lora/llama3_lora_sft.yaml
BASE_CONFIG = {
    "model_name_or_path": None,
    "adapter_name_or_path": None,
    "template": None,
    "finetuning_type": "lora",
    "trust_remote_code": True,
    ### export
    "export_dir": None,
    "export_size": 5,
    "export_device": "auto",
    "export_legacy_format": False,
}


def main(base_model_path, adapter_path, template, output_model_path):
    config = BASE_CONFIG.copy()
    config["model_name_or_path"] = base_model_path
    config["adapter_name_or_path"] = adapter_path
    config["template"] = template
    config["export_dir"] = output_model_path

    # Create a temporary YAML file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as temp_file:
        yaml.dump(config, temp_file)
        temp_config_path = temp_file.name

    try:
        # Run llamafactory-cli export command
        result = subprocess.run(
            ["llamafactory-cli", "export", temp_config_path],
            check=True,
            # capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print("Exported model")
    except subprocess.CalledProcessError as e:
        print(f"Error running llamafactory-cli: {e}")
        print(f"stderr: {e.stderr}")
        raise
    finally:
        # Clean up the temporary file
        import os

        os.unlink(temp_config_path)


@click.command()
@click.argument("base_model_path")
@click.argument("adapter_path")
@click.argument("template")
@click.argument("output_model_path")
def cli(base_model_path, adapter_path, template, output_model_path):
    main(base_model_path, adapter_path, template, output_model_path)


if __name__ == "__main__":
    cli()

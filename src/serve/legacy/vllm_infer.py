# Copied from llama-factory

# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
python -m src.eval.build_eval_data
"""

import gc
import inspect
import json
import logging
import os

from math import ceil
from pathlib import Path
from typing import Optional

import fire
import torch

from llamafactory.data.template import (
    FunctionFormatter,
    register_template,
    StringFormatter,
    ToolFormatter,
)

from ..svg_glyph_gen_v2.utils import prepare_output_dir_and_logger

# Copied from https://github.com/hiyouga/LLaMA-Factory/blob/59f2bf1ea369ca91774b99e8d94a578657be6c7c/src/llamafactory/data/template.py
try:
    register_template(
        name="qwen3_nothink",
        format_user=StringFormatter(
            slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]
        ),
        format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
        format_system=StringFormatter(
            slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]
        ),
        format_function=FunctionFormatter(
            slots=["{{content}}<|im_end|>\n"], tool_format="qwen"
        ),
        format_observation=StringFormatter(
            slots=[
                "<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            ]
        ),
        format_tools=ToolFormatter(tool_format="qwen"),
        stop_words=["<|im_end|>"],
        replace_eos=True,
    )
except Exception:
    pass

# We must use the `spawn` multiprocessing start method. Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def monkey_patch_add_metadata_column(logger=None):
    # NOTE(xk): monkey patching to save metadata
    if logger is not None:
        logger.info("monkey patching: add metadata column into dataset")

    from functools import wraps

    from llamafactory.data import converter as lf_converter, loader as lf_loader

    from llamafactory.data.converter import AlpacaDatasetConverter

    class AlpacaDatasetConverterWithMetadata(AlpacaDatasetConverter):
        def __call__(self, examples):
            output = super().__call__(examples)
            output["metadata"] = examples["metadata"]
            return output

    lf_converter.DATASET_CONVERTERS["alpaca"] = AlpacaDatasetConverterWithMetadata

    _get_preprocessed_dataset = lf_loader._get_preprocessed_dataset

    @wraps(_get_preprocessed_dataset)
    def _get_preprocessed_dataset_with_metadata(dataset, *args, **kwargs):
        new_dataset = _get_preprocessed_dataset(dataset, *args, **kwargs)
        if new_dataset is None:
            return None

        new_dataset = new_dataset.add_column("metadata", dataset["metadata"])
        return new_dataset

    lf_loader._get_preprocessed_dataset = _get_preprocessed_dataset_with_metadata


def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 8192,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    output_dir: str = "output_dir/debug",
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    max_new_tokens: int = 8192,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    default_system: Optional[str] = None,
    enable_thinking: bool = False,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    video_fps: float = 2.0,
    video_maxlen: int = 128,
    batch_size: int = 1024,
    # data parallel
    use_dp: bool = False,
    # number of data parallel processes. to divide dataset
    dp_size: int = 1,
    # rank of current data parallel process. local rank. togather with tp_size to determin CUDA_VISIBLE_DEVICESc
    local_dp_rank: int = 0,
    # rank of current data parallel process (global). to determin which chunk of dataset to load
    global_dp_rank: int = 0,
    # number of gpus to shard model weights for one data parallel process. to determin CUDA_VISIBLE_DEVICES
    tp_size: int = 1,
    # shuffle dataset for loading balance
    shuffle_dataset: bool = False,
    shuffle_seed: int = 42,
    overwrite: bool = False,
):
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """

    # set up output dir and logger
    should_skip, logger = prepare_output_dir_and_logger(
        output_dir=output_dir,
        overwrite=overwrite,
    )

    # NOTE: skip need special handling
    save_dir = Path(output_dir) / "infer"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = save_dir / f"infer-rank_{global_dp_rank}.jsonl"
    if os.path.exists(save_name) and should_skip:
        logger.info(f"File {save_name} already exists. Overwriting.")
        exit()
    with open(save_name, "w", encoding="utf-8") as f:
        pass

    if use_dp:
        if pipeline_parallel_size > 1:
            raise ValueError(
                "Pipeline parallel is no supported for data parallel. Please set pipeline_parallel_size to 1."
            )
        gpu_ids = list(range(local_dp_rank * tp_size, (local_dp_rank + 1) * tp_size))
        if gpu_ids[-1] >= torch.cuda.device_count():
            raise ValueError(
                f"tp_size {tp_size} is too large for local_dp_rank {local_dp_rank} with {torch.cuda.device_count()} gpus."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
        logger.info(
            f"Global/Local DP Rank [{global_dp_rank}/{local_dp_rank}]: "
            f"DP Size: {dp_size}, TP Size: {tp_size}, "
            f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}"
        )
    else:
        dp_size: int = 1
        local_dp_rank: int = 0
        global_dp_rank: int = 0

    # [NOTE]xk: there is some hack in llama-factory, if you change CUDA_VISIBLE_DEVICES after import,
    # then it leads to ray not init like: https://github.com/vllm-project/vllm/issues/15385
    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.extras.constants import IGNORE_INDEX
    from llamafactory.extras.misc import get_device_count
    from llamafactory.extras.packages import is_vllm_available
    from llamafactory.hparams import get_infer_args
    from llamafactory.model import load_tokenizer
    from tqdm import tqdm
    from transformers import Seq2SeqTrainingArguments

    if is_vllm_available():
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    else:
        raise ImportError("Please install vLLM to use vllm_infer.")

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            default_system=default_system,
            enable_thinking=enable_thinking,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )
    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"generating_args: {generating_args}")

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate

    # load datasets
    monkey_patch_add_metadata_column(logger)
    dataset_module = get_dataset(
        template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module
    )
    train_dataset = dataset_module["train_dataset"]
    if shuffle_dataset:
        print(f"shuffle dataset with seed {shuffle_seed}")
        train_dataset = train_dataset.shuffle(seed=shuffle_seed)

    # spit dataset into chunks for each data parallel process
    num_samples_per_rank = ceil(len(train_dataset) / dp_size)
    start = global_dp_rank * num_samples_per_rank
    end = min(start + num_samples_per_rank, len(train_dataset))
    logger.info(
        f"Global/Local DP Rank [{global_dp_rank}/{local_dp_rank}]: "
        f"Processing: ({start}, {end}) sample range. "
        f"({end-start}/{len(train_dataset)}) current/total sample."
    )

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
        "tensor_parallel_size": tp_size,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2, "audio": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    logger.info(f"engine_args: {engine_args}")
    llm = LLM(**engine_args)

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty
        or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    # Store all results in these lists
    # all_prompts, all_preds, all_labels = [], [], []

    len_written = 0
    # Add batch process to avoid the issue of too many files opened
    for i in tqdm(range(start, end, batch_size), desc="Processing batched inference"):
        vllm_inputs, prompts, labels = [], [], []
        batch = train_dataset[i : min(i + batch_size, end)]

        for j in range(len(batch["input_ids"])):
            if batch["images"][j] is not None:
                image = batch["images"][j]
                multi_modal_data = {
                    "image": template_obj.mm_plugin._regularize_images(
                        image,
                        image_max_pixels=image_max_pixels,
                        image_min_pixels=image_min_pixels,
                    )["images"]
                }
            elif batch["videos"][j] is not None:
                video = batch["videos"][j]
                multi_modal_data = {
                    "video": template_obj.mm_plugin._regularize_videos(
                        video,
                        image_max_pixels=image_max_pixels,
                        image_min_pixels=image_min_pixels,
                        video_fps=video_fps,
                        video_maxlen=video_maxlen,
                    )["videos"]
                }
            elif batch["audios"][j] is not None:
                audio = batch["audios"][j]
                audio_data = template_obj.mm_plugin._regularize_audios(
                    audio,
                    sampling_rate=16000,
                )
                multi_modal_data = {
                    "audio": zip(audio_data["audios"], audio_data["sampling_rates"])
                }
            else:
                multi_modal_data = None

            vllm_inputs.append(
                {
                    "prompt_token_ids": batch["input_ids"][j],
                    "multi_modal_data": multi_modal_data,
                }
            )
            prompts.append(
                tokenizer.decode(batch["input_ids"][j], skip_special_tokens=False)
            )
            labels.append(
                tokenizer.decode(
                    list(filter(lambda x: x != IGNORE_INDEX, batch["labels"][j])),
                    skip_special_tokens=skip_special_tokens,
                )
            )

        results = llm.generate(vllm_inputs, sampling_params, lora_request=lora_request)
        preds = [result.outputs[0].text for result in results]
        metadatas = batch["metadata"]
        len_written += len(prompts)

        # Write all results at once outside the loop
        with open(save_name, "a", encoding="utf-8", buffering=8192 * 16) as f:
            for text, pred, label, metadata in zip(prompts, preds, labels, metadatas):
                f.write(
                    json.dumps(
                        {
                            "prompt": text,
                            "predict": pred,
                            "label": label,
                            "metadata": metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        # Accumulate results
        # all_prompts.extend(prompts)
        # all_preds.extend(preds)
        # all_labels.extend(labels)
        gc.collect()

    logger.info("*" * 70)
    logger.info(
        f"{len_written} total generated results have been saved at {save_name}."
    )
    logger.info("*" * 70)


if __name__ == "__main__":
    fire.Fire(vllm_infer)

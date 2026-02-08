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

from llamafactory.data.template import (
    FunctionFormatter,
    register_template,
    StringFormatter,
    ToolFormatter,
)
from llamafactory.train.tuner import run_exp


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


def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()

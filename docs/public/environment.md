# Environment Setup

## Prerequisites

- Python 3.11 (3.10+ may work, but configs target 3.11)
- Git
- Optional: CUDA-capable GPU for training or high-throughput inference

<details><summary>`.env` file for HF and WandB; directory paths.</summary>

```bash
# WANDB_API_KEY=?
# WANDB_PROJECT=?
# WANDB_MODE=?
# WANDB_ENTITY=?

# HF_TOKEN=hf_?
HF_HOME=hf_cache/
```

```bash
mkdir -p data saves outputs misc third_party 
```
</details>

## Conda

Install conda following https://github.com/xk-huang/wiki/wiki/Start-a-new-project.


### Training environment

```bash
conda_env_name=svg_glyph_llm_train
conda env remove -n ${conda_env_name}

conda create -n ${conda_env_name} -y python=3.11
conda activate ${conda_env_name}
which python pip


# If CUDA-toolkit is not installed:
# conda install -y cuda-toolkit'<13' -c conda-forge
# !!! IMPORTENT !!!
# Run `conda activate ${conda_env_name}` again or exit current shell and restart, to make sure cuda compile env args are set.
# e.g., CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
# GXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
# CXXFLAGS=... -L$CONDA_PREFIX/targets/x86_64-linux/lib/stubs
# conda activate ${conda_env_name}
# which gcc nvcc


# Install core dependency
pip install uv
uv pip install -r requirements.txt


# Install llama-factory
uv pip install 'llamafactory[torch,metrics,deepspeed,vllm]==0.9.3' vllm==0.8.5.post1


# pip uninstall -y ninja
# uv pip install --no-cache-dir ninja
uv pip install --no-cache-dir flash-attn==2.7.2.post1 --no-build-isolation

# Eval packages
uv pip install torchmetrics==1.8.1 openai-clip==1.0.1 lpips==0.1.4

# If libcairo.so.2 shared library is missing
conda install -y cairosvg==2.8.2 -c conda-forge
```

### Eval environment

```bash
conda_env_name=svg_glyph_llm_eval
conda env remove -n ${conda_env_name}

conda create -n ${conda_env_name} -y python=3.11
conda activate ${conda_env_name}
which python pip


# If CUDA-toolkit is not installed:
# conda install -y cuda-toolkit'<13' -c conda-forge
# !!! IMPORTENT !!!
# Run `conda activate ${conda_env_name}` again or exit current shell and restart, to make sure cuda compile env args are set.
# e.g., CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
# GXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
# CXXFLAGS=... -L$CONDA_PREFIX/targets/x86_64-linux/lib/stubs
# conda activate ${conda_env_name}
# which gcc nvcc


# Install core dependency
pip install uv
uv pip install -r requirements.txt


# vllm
# Change torch-backend according to your cuda version
# Secure transformers version to avoid API incompatibility
uv pip install transformers==4.57.3 vllm==0.11.0 --torch-backend=cu128

# pip uninstall -y ninja
# uv pip install --no-cache-dir ninja
# xformers requires at most flath-attn 2.8.2
uv pip install flash-attn==2.8.2 --no-build-isolation
# Top p and top k sampling
uv pip install flashinfer-python==0.3.1.post1


# Eval packages
uv pip install torchmetrics==1.8.1 openai-clip==1.0.1 lpips==0.1.4
uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# If libcairo.so.2 shared library is missing
conda install -y cairosvg==2.8.2 -c conda-forge
```
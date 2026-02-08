#!/bin/bash
set -e



# part 1
feature install genai_conda
# conda-setup

# NOTE: sync ssh key with dotsync2: https://www.internal.example.com/wiki/Open_Source/Maintain_a_FB_OSS_Project/Devserver_GitHub_Access/#on-demand
# copy key from devserver
# mkdir -p ~/.ssh
# vim ~/.ssh/id_ed25519
# # devserve: cat ~/.ssh/id_ed25519
# # NOTE: must add +x to dirctory, otherwise we cannot access
# chmod 700 ~/.ssh
# chmod 600 ~/.ssh/id_ed25519

# install oh-my-zsh
with-proxy curl -fsSL https://raw.githubusercontent.com/xk-huang/dotfiles/main/scripts/setup_env.sh > ~/install-mydot.sh
rm -rf ${HOME}/.oh-my-zsh
SKIP_DELTA=1 ONLY_DOWNLOAD=1 with-proxy sh ~/install-mydot.sh

if [[ -f ~/.zshrc.pre-oh-my-zsh ]]; then
    rm ~/.zshrc
    mv ~/.zshrc.pre-oh-my-zsh ~/.zshrc
    echo "mv ~/.zshrc.pre-oh-my-zsh to ~/.zshrc"
else
    echo "~/.zshrc.pre-oh-my-zsh does not exist"
fi

with-proxy zsh

# clone repo
mkdir -p ~/codes
cd ~/codes
git clone git@github.com:xk-huang/svg_glyph_llm.git
code ~/codes/svg_glyph_llm

echo "Done part 1 env setup, now manually run part 2"
exit



# part 2
# Copy `cluster_scripts/.torchxconfig-template` to `.torchxconfig` and update `conda_pkg_id` if needed.
cp cluster_scripts/.torchxconfig-template .torchxconfig

if [[ -f ~/.env ]]; then
    echo "Copy ~/.env to .env"
    cp ~/.env .env
else
    echo "Create .env"
    vim .env
fi

mkdir ../svg_glyph_llm_{data,saves,misc,third_party,outputs}
ln -s ../svg_glyph_llm_data data
ln -s ../svg_glyph_llm_saves saves
ln -s ../svg_glyph_llm_misc misc
ln -s ../svg_glyph_llm_third_party third_party
ln -s ../svg_glyph_llm_outputs outputs

# mount

mkdir -p /mnt/genads_models
storagefs genads_models /mnt/genads_models

# on-demand, no permission
mkdir -p ~/mnt/genads_models
storagefs genads_models $(realpath ~/mnt/genads_models)

# unmount: fusermount3 -u /tmp/mani
echo "Done part 2 env setup"


# Directly download env from pkgcli
function validate_conda_env() {
  conda_pkg_id="$1"
  save_path="$2"
  mkdir -p "${save_path}"
  pkgcli fetch "${conda_pkg_id}" -d "${save_path}"

  conda activate ${save_path}
  which pip
  pip list | grep llamafactory || echo "llamafactory not installed"
  pip list | grep vllm || echo "vllm not installed"
  if storage_cli --help > /dev/null 2>&1 ; then
    echo "storage_cli installed"
  else
    echo "Missing storage_cli. Build with storage_cli"
  fi
}

# 251012 svg_glyph_llm_train llamafactory
conda_pkg_id=REDACTED
save_path=misc/svg_glyph_llm_train
validate_conda_env $conda_pkg_id $save_path


# 251012 svg_glyph_llm_eval vllm
conda_pkg_id=REDACTED
save_path=misc/svg_glyph_llm_eval
validate_conda_env $conda_pkg_id $save_path

# # part 3 conda training env

# # feature install genai_conda
# # conda-setup

# conda_env_name=svg_glyph_llm_train
# conda env remove -n ${conda_env_name}

# with-proxy conda create -n ${conda_env_name} -y python=3.11
# conda activate ${conda_env_name}
# which python pip

# GCC_VERSION=15.1.0
# CUDA_NVCC_VERSION=12.2.0
# with-proxy conda install -y gcc==${GCC_VERSION} -c conda-forge
# with-proxy conda install -y nvidia/label/cuda-${CUDA_NVCC_VERSION}::cuda-nvcc
# # NOTE: for deepspeed cpu adam building: curand & cudart
# with-proxy conda install -y nvidia/label/cuda-${CUDA_NVCC_VERSION}::cuda-toolkit

# which gcc nvcc

# with-proxy pip install uv
# with-proxy uv pip install -r requirements.txt


# # Install llama-factory
# # install vllm >= 0.8.4 supports qwen3moe
# with-proxy uv pip install 'llamafactory[torch,metrics,deepspeed,vllm]==0.9.3' vllm==0.8.5.post1
# # with-proxy uv pip install 'git+https://github.com/hiyouga/LLaMA-Factory.git@59f2bf1[torch,metrics,deepspeed,vllm]'
# # WARNING: the newer version leads to loss=0 & grad_norm=NaN
# # compare env w/ requirement.full.250904.txt

# # pip uninstall -y ninja
# # with-proxy uv pip install --no-cache-dir ninja
# with-proxy uv pip install --no-cache-dir flash-attn==2.7.2.post1 --no-build-isolation
# # with-proxy uv pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation
# # with-proxy uv pip install --no-cache-dir flashinfer_python==0.2.2.post1 --no-build-isolation


# # Eval packages
# with-proxy uv pip install torchmetrics==1.8.1 openai-clip==1.0.1 lpips==0.1.4


# # Other tools
# with-proxy uv pip install gpustat
# # with-proxy yarn global add svgo@4.0.0
# # CLUSTER node does not have libcairo.so.2 shared library
# with-proxy conda install -y cairosvg==2.8.2
# echo "Done part 3 env setup"



# # part 4 conda eval env
# # feature install genai_conda
# # conda-setup

# conda_env_name=svg_glyph_llm_eval
# conda env remove -n ${conda_env_name}

# with-proxy conda create -n ${conda_env_name} -y python=3.11
# conda activate ${conda_env_name}
# which python pip
# with-proxy pip install uv

# # cuda-toolkit 12.9.1
# # cuda driver compatibility: https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html
# with-proxy conda install -y cuda-toolkit'<13' -c conda-forge
# which gcc nvcc


# with-proxy uv pip install -r requirements.txt

# # NOTE: We prefer vllm for its reliability
# # with-proxy uv pip install sglang==0.5.3rc0
# # This commit fix port issue on On Demand
# # git clone --branch fix_find_port --single-branch --depth 1 https://github.com/xk-huang/sglang.git third_party/sglang
# # with-proxy uv pip install -e third_party/sglang/python


# # !!! IMPORTENT !!!
# # Run `conda activate ${conda_env_name}` again or exit current shell and restart, to make sure cuda compile env args are set.
# # e.g., CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
# # GXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
# # CXXFLAGS=... -L$CONDA_PREFIX/targets/x86_64-linux/lib/stubs
# conda activate ${conda_env_name}
# which gcc nvcc

# # SGLang requires the below building path `*/lib64`.
# # But it should be mandatory: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#environment-setup
# if [ ! -d "$CONDA_PREFIX/lib64" ]; then
#     echo "Creating symlink from lib to lib64 in ${CONDA_PREFIX}"
#     ln -s "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"
# else
#     echo "lib64 already exists in ${CONDA_PREFIX}"
# fi
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# # vllm
# with-proxy uv pip install vllm==0.11.0 --torch-backend=cu129
# # to support qwen3-vl
# with-proxy uv pip install ninja
# with-proxy uv pip install flash-attn==2.8.2 --no-build-isolation
# # xformers requires at most flath-attn 2.8.2
# with-proxy uv pip install flashinfer-python==0.3.1.post1
# # for top p and top k sampling

# # Eval packages
# with-proxy uv pip install torchmetrics==1.8.1 openai-clip==1.0.1 lpips==0.1.4
# with-proxy uv pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation

# # tool
# with-proxy uv pip install gpustat
# # CLUSTER node does not have libcairo.so.2 shared library
# with-proxy conda install -y cairosvg==2.8.2
# echo "Done part 4 env setup"

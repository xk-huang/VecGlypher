# ─────────────────────────────────────────────────────────────
# File: submit.py
# Usage examples:
#   pip install hydra-core omegaconf
#   python scripts/sft_on_cluster_submitter/submit.py dry_run=true local_run=true # run all jobs in job_args (sequentially)
#   python scripts/sft_on_cluster_submitter/submit.py jobs=char                   # run only one entry
#   python scripts/sft_on_cluster_submitter/submit.py jobs=char,word dry_run=true # print commands only
#   python scripts/sft_on_cluster_submitter/submit.py base_args._cluster_param.nproc_per_node=8     # override any field at CLI

# How to: check the config file scripts/sft_on_cluster_submitter/conf/config.yaml.
# 1. `_*` arg will be discarded, while other args will be appended as `args1=var1 args2=var2 ...` format.
# 2. `_output_base_dir` is the output dir prefix, which is used to build `output_dir`.
# 3. `_exp_name` and `_job_name` is the experiment and the job names,
#     which is used to build `output_dir=<_output_base_dir>/<_exp_name>/<_job_name>`.
#     and the CLUSTER run name`--name=<_exp_name>-<_job_name>-<_cluster_timestamp>`.
# 4. `dry_run` will print the commands only. For the generated shell scripts,
#     they only build the env and does not submit the job.
# 5. `local_run` use `torchrun` instead of `torchx` to run locally.
# 6. `scheduler_args` to comma separated key=value pairs to define CLUSTER scheduler.
# 7. `host` to specify the host type, e.g., tc_any, gpu_80g_pool, tc_any_40g.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations

import copy
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep merge two dict-like objects. If override has a key set to None, the key is removed.
    Lists are replaced (not merged)."""

    def _merge(a, b):
        if b is None:
            return None  # signal deletion at parent level
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            out = dict(a)
            for k, v in b.items():
                if v is None:
                    out.pop(k, None)
                else:
                    out[k] = _merge(a.get(k), v) if k in a else copy.deepcopy(v)
            return out
        else:
            return copy.deepcopy(b)

    merged = _merge(base, override)
    return {} if merged is None else merged


def coerce_cli_value(v: Any) -> str:
    """Convert Python values into CLI "key=value" friendly strings."""
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    if isinstance(v, bool):
        return str(v).lower()
    if v is None:
        return ""
    return str(v)


def args_to_cli(args: Mapping[str, Any]) -> List[str]:
    # NOTE(xk): keys starts with "_" are ignored, e.g., "_job_name"
    cli = []
    for k, v in args.items():
        if k.startswith("_"):
            continue
        if v is None:
            continue  # explicit deletion
        s = coerce_cli_value(v)
        if s == "":
            continue
        cli.append(f"{k}={s}")
    return cli


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def materialize_and_save(
    exp_name: str,
    cfg: DictConfig,
    merged_args: Mapping[str, Any],
    torchx_cmd: List[str],
) -> None:
    # Try to derive an output dir from args if present
    artifact_base_dir = Path(cfg.meta.artifact_root)
    artifact_out_dir = artifact_base_dir / f"{cfg.meta.run_id}-{exp_name}"
    ensure_dir(artifact_out_dir)
    dump_parsed_config(artifact_out_dir, cfg, merged_args, torchx_cmd)

    out_dir = merged_args.get("output_dir")
    if out_dir is not None:
        out_dir = Path(str(out_dir))
        ensure_dir(out_dir)
        dump_parsed_config(out_dir, cfg, merged_args, torchx_cmd)


def dump_parsed_config(out_dir, cfg, merged_args, torchx_cmd):
    # Save fully resolved config snapshot
    (out_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    (out_dir / "resolved_args.json").write_text(
        json.dumps(merged_args, indent=2, ensure_ascii=False)
    )
    (out_dir / "launch_cmd.sh").write_text(
        " \\\n".join(shlex.quote(x) for x in torchx_cmd)
    )
    print(f"[info] Saved resolved args to: {out_dir / 'resolved_args.json'}")
    print(f"[info] Saved launch cmd args to: {out_dir / 'launch_cmd.sh'}")


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig = None) -> None:
    # Allow rich interpolations like ${now:...} and env vars
    OmegaConf.resolve(cfg)

    # NOTE: the output base dir is the name of the config file
    cfg_name = HydraConfig.get().job.config_name
    if cfg.base_args.get("_exp_name", None) is None:
        cfg.base_args._exp_name = cfg_name
        print(
            f"[info] no exp_name found in base_args. Using config name as exp_name: {cfg_name}"
        )

    # Determine which jobs to run
    if cfg.jobs == "ALL" or cfg.jobs is None:
        job_args = cfg.get("job_args", None)
        if job_args is None:
            # NOTE: default, a job using default base_args,
            # will not be used if cfg.job_args is set.
            OmegaConf.set_struct(cfg, False)
            print("[info] No default job args found. Using empty dict")
            cfg.job_args = {}
            cfg.job_args["default_job"] = {}
            OmegaConf.set_struct(cfg, True)
            job_names = ["default_job"]
        else:
            job_names = list(cfg.job_args.keys())
    elif isinstance(cfg.jobs, (list, tuple, ListConfig)):
        job_names = list(cfg.jobs)
    else:
        job_names = [x.strip() for x in str(cfg.jobs).split(",") if x.strip()]

    missing = [e for e in job_names if e not in cfg.job_args]
    if missing:
        print(
            f"Unknown experiment(s) in jobs=: {missing} Valid: {list(cfg.job_args.keys())}",
            file=sys.stderr,
        )
        sys.exit(2)

    launched = []
    for idx, job_name in enumerate(job_names):
        print(f"[info] Start job: {job_name}")
        # Compose context for this experiment so ${exp_name} can be used in YAML
        ctx = {
            "_exp_index": idx,
            "_run_id": cfg.meta.run_id,
        }
        # Merge and resolve base_args + job_args[name]
        base_args = OmegaConf.create(cfg.base_args)
        exp_over = OmegaConf.create(cfg.job_args.get(job_name, {}))
        if exp_over.get("_job_name", None) is None:
            exp_over._job_name = job_name
            print(f"[warning] no job_name found in job_args. Using {job_name}")

        merged_args = OmegaConf.merge(base_args, exp_over, ctx)

        if cfg.base_args.output_dir is not None:
            raise ValueError("base_args.output_dir should be None.")
        merged_args = prepare_outupt_dir(merged_args)
        torchx_prefix = build_cluster_param_prefix(merged_args, cfg.local_run)

        OmegaConf.resolve(merged_args)
        merged_args = OmegaConf.to_container(merged_args, resolve=True)

        # Build the final torchx command
        cli_kvs = args_to_cli(merged_args)
        cmd = [*torchx_prefix, *cli_kvs]

        # Save snapshots
        try:
            materialize_and_save(job_name, cfg, merged_args, cmd)
        except Exception as e:
            print(
                f"[warn] could not save resolved config for {job_name}: {e}",
                file=sys.stderr,
            )

        # print(
        #     f"[torchx command: {job_name}] "
        #     + " \\\n".join(shlex.quote(x) for x in cmd)
        # )
        launched.append((job_name, cmd))

        if not cfg.dry_run:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(
                    f"[error] {job_name} failed with exit code {e.returncode}",
                    file=sys.stderr,
                )
                if cfg.fail_fast:
                    sys.exit(e.returncode)
            finally:
                print(f"[info] Finished job: {job_name}")

    if cfg.dry_run:
        print(f"Dry run complete. {len(launched)} command(s) shown.")


def build_cluster_param_prefix(job_cfg, local_run=False):
    # Build static pieces of the torchx command
    job_cluster_param_cfg = job_cfg._cluster_param

    if local_run:
        torchx_prefix = [
            "NCCL_DEBUG=WARN",  # suppress NCCL debug logs, which are very verbose
            "torchrun",
            "--nnodes",
            str(job_cluster_param_cfg.nnodes),
            "--nproc-per-node",
            str(job_cluster_param_cfg.nproc_per_node),
            "--no-python",
            "--standalone",
            str(job_cluster_param_cfg.script),
        ]
    else:
        torchx_prefix = [
            "torchx",
            "run",
        ]
        scheduler = job_cluster_param_cfg.scheduler

        if scheduler is not None:
            torchx_prefix += ["--scheduler", str(scheduler)]
        if job_cluster_param_cfg.scheduler_args is not None:
            torchx_prefix += ["--scheduler_args", str(job_cluster_param_cfg.scheduler_args)]
        if job_cluster_param_cfg.dry_run is True:
            torchx_prefix += ["--dryrun"]

        torchx_prefix += [
            job_cluster_param_cfg.app,
            "--h",
            str(job_cluster_param_cfg.host),
            "--nnodes",
            str(job_cluster_param_cfg.nnodes),
            "--nproc_per_node",
            str(job_cluster_param_cfg.nproc_per_node),
            "--max_retries",
            str(job_cluster_param_cfg.max_retries),
            "--script",
            str(job_cluster_param_cfg.script),
        ]
        name = job_cluster_param_cfg.get("name", None)
        if name is None:
            now = datetime.now().isoformat().replace(":", "_").replace(".", "_")
            for k in ["_exp_name", "_job_name"]:
                if job_cfg[k] is None:
                    raise ValueError(f"Missing required key: {k} when name is None")

            name = f"{job_cfg._exp_name}-{job_cfg._job_name}-{now}"
        torchx_prefix += ["--name", name]

    # Build program args
    program = job_cluster_param_cfg.program
    if OmegaConf.is_list(program):
        torchx_prefix.append("--")
        torchx_prefix.extend(program)
    elif OmegaConf.is_dict(program):
        raise ValueError(f"program (`{program}`) cannot be a dict.")
    else:
        torchx_prefix.extend(["--", str(program)])

    config_file = job_cluster_param_cfg.get("config_file", None)
    if config_file is not None:
        torchx_prefix += [str(config_file)]
    return torchx_prefix


def prepare_outupt_dir(cfg):
    output_dir = cfg["output_dir"]
    if output_dir is not None:
        return cfg

    output_base_dir = cfg["_output_base_dir"]
    exp_name = cfg["_exp_name"]
    job_name = cfg["_job_name"]
    for k in ["_exp_name", "_job_name", "_output_base_dir"]:
        if cfg[k] is None:
            raise ValueError(f"Missing required key: {k} when output_dir is None")

    output_dir = os.path.join(output_base_dir, exp_name, job_name)
    cfg["output_dir"] = output_dir
    return cfg


if __name__ == "__main__":
    main()

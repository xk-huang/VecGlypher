"""
input_dir=outputs/250919-eval-gfont-baselines/google/gemma-3-27b-it/t_0_0
python -m src.eval.score_point_cloud_eval \
    --input_svg_dir ${input_dir}/infer \
    --output_dir ${input_dir}/results_point_cloud_eval

input_dir=outputs/250919-eval-gfont-baselines/google/gemma-3-27b-it/t_0_0
python -m src.eval.score_point_cloud_eval \
    --input_svg_dir ${input_dir}/infer \
    --output_dir ${input_dir}/results_point_cloud_eval-align_pcd \
    --align_pcd

input_dir=outputs/250919-eval-gfont-baselines/google/gemma-3-27b-it/t_0_0
python -m src.eval.score_point_cloud_eval \
    --input_svg_dir ${input_dir}/infer \
    --output_dir ${input_dir}/results_point_cloud_eval-align_pcd-estimate_scale \
    --align_pcd \
    --estimate_scale
"""

import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import torch
import tqdm
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import iterative_closest_point
from src.svg_glyph_gen_v2.svg_simplifier import SVGSimplifier
from src.svg_glyph_gen_v2.utils import (
    count_lines,
    load_jsonl_by_generator,
    prepare_output_dir_and_logger,
)
from svg.path import parse_path as parse_path_v2
from svgpathtools import svgstr2paths


def add_width_height_100(svg):
    return svg.replace(
        '<svg xmlns="http://www.w3.org/2000/svg"',
        '<svg xmlns="http://www.w3.org/2000/svg"  width="100" height="100"',
        1,
    )


def sample_points_on_svg_path(
    path_or_d: str,
    spacing: Optional[float] = None,
    num_sampled_points: Optional[int] = None,
    include_start: bool = True,
    include_end: bool = True,
) -> List[Tuple[float, float]]:
    """
    Sample points at (approximately) uniform arc-length spacing over the entire path.
    Works with lines, quadratic/cubic beziers, and arcs in the 'd' attribute.

    Parameters
    ----------
    path_or_d : str
        The SVG path 'd' string.
    spacing : float
        Desired distance between consecutive samples in the same units as the SVG.
    include_start : bool
        Include the very first point on the path (s=0).
    include_end : bool
        Include the final point (s=total_length).

    Returns
    -------
    List[(x, y)]
        Sampled points as (x, y) tuples.
    """
    if spacing is None and num_sampled_points is None:
        raise ValueError("Must specify either spacing or num_sampled_points")
    if spacing is not None and num_sampled_points is not None:
        raise ValueError("Cannot specify both spacing and num_sampled_points")

    if isinstance(path_or_d, str):
        path = parse_path_v2(path_or_d)
    else:
        path = path_or_d
    total_len = path.length(error=1e-6)

    if total_len == 0:
        # Degenerate path
        # z = path.point(0j)
        # return [(z.real, z.imag)]
        return [(0.0, 0.0)]

    # Build list of arclengths at which to sample
    if num_sampled_points is not None:
        if num_sampled_points < 2:
            raise ValueError("num_sampled_points must be >= 2")
        s_vals = np.linspace(0.0, total_len, num_sampled_points)
    else:
        if spacing <= 0:
            raise ValueError("spacing must be > 0")
        s_vals = np.arange(0.0, total_len + 1e-9, spacing)

    # Respect flags
    if not include_start and len(s_vals) and np.isclose(s_vals[0], 0.0):
        s_vals = s_vals[1:]
    if include_end and (len(s_vals) == 0 or not np.isclose(s_vals[-1], total_len)):
        s_vals = np.append(s_vals, total_len)
    if not include_end and len(s_vals) and np.isclose(s_vals[-1], total_len):
        s_vals = s_vals[:-1]

    # Invert arclength -> parameter t over the *whole* path
    # Path.ilength(s) returns complex parameter where real part is segment index + local t,
    # but path.point() expects a complex t on [0, N] with integer parts selecting segment.
    pts = []
    for s in s_vals:
        try:
            # t = path.ilength(s, error=1e-6)  # complex parameter along entire Path
            z = path.point(s / total_len)  # complex point on Path
        except Exception as e:
            raise e
        pts.append((z.real, z.imag))
    return pts


def sample_points_on_svg(
    svg,
    spacing: Optional[float] = None,
    num_sampled_points: Optional[int] = None,
    include_start: bool = True,
    include_end: bool = True,
    logger=None,
):
    paths, _ = svgstr2paths(svg)
    points = []
    for path in paths:
        try:
            path = parse_path_v2(path.d())
        except Exception as e:
            if logger is not None:
                logger.error(
                    f"Failed to parse path: `{path}` due to {e}\nsvg: `{svg}`\nWe skip this path"
                )
            continue

        points.extend(
            sample_points_on_svg_path(
                path, spacing, num_sampled_points, include_start, include_end
            )
        )
    return points


def normalize_point_cloud(pc):
    # pc: (N, 3) numpy array
    mins = pc.min(dim=0)[0]
    maxs = pc.max(dim=0)[0]
    center = (maxs + mins) / 2
    pc = pc - center
    scale = (maxs - mins).max() / 2
    pc = pc / scale
    return pc


svg_simplifer = SVGSimplifier()


def get_processed_sample(
    data,
    spacing: Optional[float] = None,
    num_sampled_points: Optional[int] = None,
    include_start: bool = True,
    include_end: bool = True,
    dtype=torch.float32,
    logger=None,
):
    predict_svg = data["predict"]
    label_svg = data["label"]
    predict_svg = svg_simplifer.decode(predict_svg)
    label_svg = svg_simplifer.decode(label_svg)
    return {
        "predict": get_normalized_pcd(
            predict_svg,
            spacing,
            num_sampled_points,
            include_start,
            include_end,
            dtype=dtype,
            logger=logger,
        ),
        "label": get_normalized_pcd(
            label_svg,
            spacing,
            num_sampled_points,
            include_start,
            include_end,
            dtype=dtype,
            logger=logger,
        ),
    }


def get_normalized_pcd(
    svg,
    spacing: Optional[float] = None,
    num_sampled_points: Optional[int] = None,
    include_start: bool = True,
    include_end: bool = True,
    dtype=torch.float32,
    logger=None,
):
    pcd = sample_points_on_svg(
        svg, spacing, num_sampled_points, include_start, include_end, logger=logger
    )
    pcd = torch.tensor(pcd, dtype=dtype)

    failed_to_norm = False
    try:
        pcd = normalize_point_cloud(pcd)
    except Exception:
        failed_to_norm = True

    if failed_to_norm or pcd.isnan().any():
        pcd = torch.zeros((1, 2), dtype=dtype)

    if len(pcd) != num_sampled_points:
        pcd = torch.zeros((num_sampled_points, 2), dtype=dtype)

    return pcd


def extend_batch_dim_if_needed(x):
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 2:
        x = x.unsqueeze(0)

    return x


def align_and_compute_chamfer_distance(
    predict_pcd,
    label_pcd,
    align_pcd=False,
    estimate_scale=False,
    logger=None,
):
    """
    predict_pcd: (N, P, 2)
    label_pcd: (N, Q, 2)
    """
    if align_pcd:
        icp_solution = iterative_closest_point(
            predict_pcd, label_pcd, estimate_scale=estimate_scale
        )
        if icp_solution.converged is False:
            if logger is not None:
                logger.warning("ICP did not converge")
            aligned_predict_pcd = predict_pcd
        else:
            aligned_predict_pcd = icp_solution.Xt

    else:
        icp_solution = None
        aligned_predict_pcd = predict_pcd

    cd_loss = chamfer_distance(
        aligned_predict_pcd,
        label_pcd,
        batch_reduction=None,
    )

    return cd_loss, icp_solution


@click.command()
@click.option("--input_svg_dir", type=click.Path(exists=True), required=True)
@click.option("--output_dir", type=click.Path(), required=True)
@click.option("--overwrite", is_flag=True, default=False)
@click.option("--num_sampled_points", type=int, default=200)
@click.option("--align_pcd/--no_align_pcd", default=False, help="Case sensitive")
@click.option(
    "--estimate_scale/--no_estimate_scale", default=False, help="Case sensitive"
)
@click.option("--num_workers", type=int, default=40)
@click.option("--batch_size", type=int, default=64)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda"]),
    default="cuda",
    help="Use cuda if align_pcd is True, otherwise it is slow.",
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    should_skip, logger = prepare_output_dir_and_logger(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    logger.info(f"args: {args}")
    if should_skip:
        exit()
    output_dir = Path(args.output_dir)

    input_svg_dir = args.input_svg_dir
    num_sampled_points = args.num_sampled_points
    num_samples = count_lines(input_svg_dir)
    logger.info(f"Number of samples: {num_samples}")

    kwargs = {
        "align_pcd": args.align_pcd,
        "estimate_scale": args.estimate_scale,
    }

    # Test load one sample
    test_load_one_sample(input_svg_dir, num_sampled_points, kwargs)

    # For-loop test
    # for_loop_process(input_svg_dir, num_samples, num_sampled_points, kwargs)

    # Dataloader test
    eval_dataset = EvalDataset(
        input_svg_dir, num_sampled_points=num_sampled_points, logger=logger
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    pbar = tqdm.tqdm(total=num_samples)
    num_processed = 0
    all_loss_list = []
    device = args.device
    for idx, processed_sample in enumerate(eval_dataloader):
        predict_pcd = processed_sample["predict"].to(device=device, non_blocking=True)
        label_pcd = processed_sample["label"].to(device=device, non_blocking=True)

        output = align_and_compute_chamfer_distance(predict_pcd, label_pcd, **kwargs)
        loss = output[0][0]
        all_loss_list.append(loss)

        pbar.update(len(predict_pcd))
        num_processed += len(predict_pcd)

    logger.info(f"Number of processed samples: {num_processed}")
    all_loss = torch.cat(all_loss_list)
    mean_loss = torch.mean(all_loss)

    suffix = ""
    for k, v in kwargs.items():
        if v is True:
            suffix += f"-{k}"

    score_stats = {
        f"chamfer_distance": mean_loss.item(),
        "total_in": num_samples,
        "total_out": num_processed,
    }
    out_stats_path = output_dir / f"score_stats{suffix}.json"
    with open(out_stats_path, "w") as f:
        json.dump(score_stats, f, ensure_ascii=False, indent=4)
    logger.info(f"Stats: {score_stats}")

    df = pd.DataFrame(all_loss.cpu(), columns=[f"chamfer_distance"])
    output_df_path = output_dir / f"score.csv"
    df.to_csv(output_df_path, index=False)
    logger.info(f"Save Score: {output_df_path}")


def for_loop_process(input_svg_dir, num_samples, num_sampled_points, kwargs):
    loss_list = []
    for _, sample in enumerate(
        tqdm.tqdm(load_jsonl_by_generator(input_svg_dir), total=num_samples)
    ):
        processed_sample = get_processed_sample(
            sample, num_sampled_points=num_sampled_points
        )
        predict_pcd = processed_sample["predict"]
        predict_pcd = extend_batch_dim_if_needed(predict_pcd)
        label_pcd = processed_sample["label"]
        label_pcd = extend_batch_dim_if_needed(label_pcd)
        output = align_and_compute_chamfer_distance(predict_pcd, label_pcd, **kwargs)
        loss = output[0][0]
        loss_list.append(loss)
    return loss_list


def test_load_one_sample(input_svg_dir, num_sampled_points, kwargs):
    sample = next(iter(load_jsonl_by_generator(input_svg_dir)))
    processed_sample = get_processed_sample(
        sample, num_sampled_points=num_sampled_points
    )

    predict_pcd = processed_sample["predict"]
    predict_pcd = extend_batch_dim_if_needed(predict_pcd)
    label_pcd = processed_sample["label"]
    label_pcd = extend_batch_dim_if_needed(label_pcd)
    output = align_and_compute_chamfer_distance(predict_pcd, label_pcd, **kwargs)
    # output type: Tuple[Tuple[torch.Tensor, Optional[torch.Tensor]], Optional[ICPSolution]]
    loss = output[0][0]


class EvalDataset(torch.utils.data.IterableDataset):
    def __init__(self, input_svg_dir, num_sampled_points=200, logger=None):
        self.input_svg_dir = input_svg_dir
        self.num_sampled_points = num_sampled_points
        self.logger = logger

    def __iter__(self):
        # IMPORTANT: create the generator *inside* __iter__ so each worker gets its own.
        data_iter = load_jsonl_by_generator(self.input_svg_dir)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process data loading
            iter_ = data_iter
        else:
            # NOTE: Be aware of the worker distribution.
            # Multi-process data loading: shard the stream by worker_id
            # Each worker takes every Nth element starting from its id.
            # This avoids overlap without needing to know dataset length.
            iter_ = itertools.islice(
                data_iter, worker_info.id, None, worker_info.num_workers
            )

        for sample in iter_:
            processed_sample = get_processed_sample(
                sample, num_sampled_points=self.num_sampled_points, logger=self.logger
            )
            yield processed_sample


if __name__ == "__main__":
    main()

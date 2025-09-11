#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt

# ------------------ helpers ------------------

def list_tasks(h5):
    """列出 /tasks 下可用数据集名称"""
    tasks = []
    if "tasks" in h5:
        for k, v in h5["tasks"].items():
            if isinstance(v, h5py.Dataset):
                tasks.append(k)
    return tasks

def get_time_array(dset):
    """
    尝试拿时间坐标。Dedalus 的时间一般挂在第0维的维度标尺上（可能带 hash 名）。
    取不到就返回 None，不影响绘图。
    """
    try:
        dim0 = dset.dims[0]
        # 有些文件会挂多个scale，只取第一个
        if len(dim0) > 0:
            sc = dim0[0]
            return np.array(sc[...]).astype(float)
    except Exception:
        pass
    # 兜底：尝试从 /scales 下找名字里带 "sim_time" 的
    try:
        f = dset.file
        if "scales" in f:
            for name, node in f["scales"].items():
                if "time" in name.lower() or "sim" in name.lower():
                    return np.array(node[...]).astype(float)
    except Exception:
        pass
    return None

def reduce_to_2d(arr, prefer='magnitude', component=None):
    """
    把任意形状的切片数据变成 2D：
    1) 先 squeeze 去掉长度为1的轴；
    2) 如果还剩 3 维，优先把“分量轴”(长度为2或3)按 'magnitude' 求范数，
       或按 component（x/y/z/0/1/2）取对应分量。
    """
    a = np.squeeze(np.asarray(arr))
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        # 识别分量轴（长度为2或3）
        comp_axes = [i for i, s in enumerate(a.shape) if s in (2, 3)]
        if comp_axes:
            ax = comp_axes[-1]  # 取最后一个更常见
            if component is not None:
                idx = parse_component(component)
                if idx >= a.shape[ax]:
                    raise ValueError(f"component={component} 超出范围（该轴长度为 {a.shape[ax]}）")
                return np.take(a, idx, axis=ax)
            # 默认取模
            if prefer == 'magnitude':
                return np.linalg.norm(a, axis=ax)
            else:
                # 明确指明 prefer='component' 但没给 component
                raise ValueError("检测到矢量场；请提供 --component 或使用 --magnitude")
        else:
            # 没有明显的矢量分量轴，尝试把长度>1的三轴里挑两个最大维用于图像
            # 这里保守处理：如果存在任意长度为1的轴已被 squeeze，不该出现这种情况
            # 仍然抛错，提示用户检查数据
            raise ValueError(f"期望 2D 或 (2D+分量) 数据，但得到形状 {a.shape}")
    raise ValueError(f"期望 2D 数据，得到 {a.ndim} 维。原始形状：{arr.shape}")

def parse_component(comp):
    """解析 --component（'x'/'y'/'z' 或 0/1/2）"""
    if isinstance(comp, str):
        m = {'x': 0, 'y': 1, 'z': 2}
        if comp.lower() in m:
            return m[comp.lower()]
        try:
            return int(comp)
        except Exception:
            pass
    if isinstance(comp, (int, np.integer)):
        return int(comp)
    raise ValueError("component 只支持 x/y/z 或 0/1/2")

def get_extent_from_scales(dset, sample_frame=0):
    """
    尝试用维度标尺构造 imshow 的 extent=[xmin,xmax,ymin,ymax]。
    忽略 time 轴和长度为1的轴；顺序按 (rows, cols) -> (y, x)。
    获取失败则返回 None（让 imshow 用像素索引）。
    """
    try:
        # 先取一个样本帧，找到被 squeeze 后的二维空间轴索引
        sample = np.squeeze(dset[sample_frame, ...])
        # 找出原数据中 time 之后的空间轴，以及其中非单例的两个轴
        spatial_axes = list(range(1, dset.ndim))
        non_single = [ax for ax in spatial_axes if dset.shape[ax] != 1]
        if len(non_single) < 2:
            return None  # 不足两维，不设 extent

        # 取前两个非单例轴，构造成 (y_axis, x_axis)
        y_ax, x_ax = non_single[0], non_single[1]

        def axis_coords(ax):
            dim = dset.dims[ax]
            if len(dim) > 0:
                sc = dim[0]
                coords = np.array(sc[...]).astype(float)
                return coords.min(), coords.max()
            else:
                n = dset.shape[ax]
                return 0.0, float(n - 1)

        ymin, ymax = axis_coords(y_ax)
        xmin, xmax = axis_coords(x_ax)
        return [xmin, xmax, ymin, ymax]
    except Exception:
        return None

def compute_global_vmin_vmax(dset, frames, reducer_kwargs):
    vmin, vmax = np.inf, -np.inf
    for i in frames:
        data2d = reduce_to_2d(dset[i, ...], **reducer_kwargs)
        cur_min = np.nanmin(data2d)
        cur_max = np.nanmax(data2d)
        if cur_min < vmin: vmin = cur_min
        if cur_max > vmax: vmax = cur_max
    return float(vmin), float(vmax)

# ------------------ main ------------------

def main(args):
    fpath = Path(args.file)
    if not fpath.exists():
        raise FileNotFoundError(f"未找到文件：{fpath}")

    with h5py.File(fpath, "r") as h5:
        tasks = list_tasks(h5)
        if not tasks:
            raise RuntimeError("文件中未找到 /tasks 下的数据集")

        task = args.task or tasks[0]
        if task not in tasks:
            raise ValueError(f"任务 {task} 不在文件中。可选：{tasks}")

        dset = h5["tasks"][task]
        nframes = dset.shape[0]
        time_arr = get_time_array(dset)

        # 帧范围
        start = 0 if args.start is None else max(0, args.start)
        stop = nframes - 1 if args.stop is None else min(nframes - 1, args.stop)
        step = args.step
        frames = list(range(start, stop + 1, step))
        if not frames:
            raise ValueError("帧范围为空，请检查 --start/--stop/--step")

        # 矢量处理参数
        reducer_kwargs = {
            "prefer": "magnitude" if args.magnitude else "component",
            "component": args.component,
        }

        # extent（物理坐标可视化）
        extent = None if args.no_extent else get_extent_from_scales(dset)

        # 全局 vmin/vmax
        if args.auto_vrange:
            vmin, vmax = compute_global_vmin_vmax(dset, frames, reducer_kwargs)
            print(f"Global vmin/vmax = {vmin:.6g} / {vmax:.6g}")
        else:
            vmin = args.vmin
            vmax = args.vmax

        # 输出目录
        outdir = Path(args.outdir) if args.outdir else None
        if outdir:
            outdir.mkdir(parents=True, exist_ok=True)

        # 单帧预览或批量导出
        for i in frames:
            img = reduce_to_2d(dset[i, ...], **reducer_kwargs)
            fig, ax = plt.subplots(figsize=(6, 4.8), dpi=args.dpi)
            im = ax.imshow(
                img,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                vmin=vmin, vmax=vmax,
                cmap=args.cmap
            )
            cbar = plt.colorbar(im, ax=ax)
            # colorbar label
            units = dset.attrs.get("units", "").decode() if isinstance(dset.attrs.get("units", ""), bytes) else dset.attrs.get("units", "")
            if units:
                cbar.set_label(units)
            # title
            if time_arr is not None and i < len(time_arr):
                ax.set_title(f"{task} | frame {i} | t = {time_arr[i]:.6g}")
            else:
                ax.set_title(f"{task} | frame {i}")

            if extent is not None:
                im.set_extent(extent)
                ax.set_xlabel("x")
                ax.set_ylabel("y")

            plt.tight_layout()

            if outdir:
                png = outdir / f"{task}_{i:05d}.png"
                plt.savefig(png, bbox_inches="tight", dpi=args.dpi)
                plt.close(fig)
            else:
                # 只显示第一帧以免卡屏
                plt.show()
                plt.close(fig)
                break  # 没有 outdir 就预览一帧后退出

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot 2D snapshots from Dedalus HDF5.")
    p.add_argument("--file", default="snapshots_channel_2D_s1.h5", help="HDF5 文件路径")
    p.add_argument("--task", default=None, help="要绘制的 /tasks/<name> 数据集（默认取第一个）")
    p.add_argument("--start", type=int, default=None, help="起始帧索引（含）")
    p.add_argument("--stop", type=int, default=None, help="结束帧索引（含）")
    p.add_argument("--step", type=int, default=1, help="帧步长")
    p.add_argument("--magnitude", action="store_true", help="矢量场取模（默认如果检测到矢量且未指定 component，会使用取模）")
    p.add_argument("--component", default=None, help="矢量场取某一分量（x/y/z 或 0/1/2）")
    p.add_argument("--auto-vrange", action="store_true", help="自动扫描选定帧，计算全局 vmin/vmax")
    p.add_argument("--vmin", type=float, default=None, help="手动设置 vmin")
    p.add_argument("--vmax", type=float, default=None, help="手动设置 vmax")
    p.add_argument("--no-extent", action="store_true", help="不从标尺推断物理坐标 extent（用像素坐标）")
    p.add_argument("--cmap", default="viridis", help="Matplotlib colormap 名称（如 viridis, plasma, turbo 等）")
    p.add_argument("--dpi", type=int, default=150, help="保存/显示的 DPI")
    p.add_argument("--outdir", default=None, help="输出 PNG 的目录；不提供则仅预览一帧后退出")
    args = p.parse_args()
    main(args)

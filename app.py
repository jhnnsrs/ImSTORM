#!/usr/bin/env python3
"""
STORM simulation, localization & Arkitekt datasource/service
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import tifffile as tif
import xarray as xr

# ───────────────────────── Arkitekt & Mikro ──────────────────────────
from arkitekt_next import easy, register, progress
from mikro_next.api.schema import (
    Image,
    from_array_like,
    create_stage,
    PartialAffineTransformationViewInput,
    ColorMap,
)

from microEye.Filters import BandpassFilter
from microEye.fitting.fit import CV_BlobDetector, localize_frame
from microEye.fitting.results import FittingMethod

# ───────────────────────── simulation parameters ──────────────────────
N_FRAMES: int = 50
FRAME_SHAPE: tuple[int, int] = (256, 256)
SPOTS_PER_FRAME: tuple[int, int] = (5, 15)
SIGMA_PSF: float = 1.5
PHOTON_COUNT: tuple[int, int] = (500, 2000)
BG_LEVEL: int = 100
NOISE_STD: float = 10
REL_THRESHOLD: float = 0.4
ROI_SIZE: int = 13

# -----------------------------------------------------------------------------
# Arkitekt service setup
# -----------------------------------------------------------------------------
app = easy("STORM_Reconstruction_Service")

_pre_filter = BandpassFilter()
_peak_detector = CV_BlobDetector()

# ───────────────────────── helper functions ───────────────────────────

def _gaussian_spot(shape: tuple[int, int], y0: float, x0: float, sigma: float, amp: float):
    y = np.arange(shape[0])[:, None]
    x = np.arange(shape[1])[None, :]
    return amp * np.exp(-((y - y0) ** 2 + (x - x0) ** 2) / (2 * sigma ** 2))


def _simulate_frame(shape: tuple[int, int], n_spots: int) -> np.ndarray:
    frame = np.full(shape, BG_LEVEL, dtype=np.float32)
    ys = np.random.uniform(0, shape[0], n_spots)
    xs = np.random.uniform(0, shape[1], n_spots)
    amps = np.random.uniform(*PHOTON_COUNT, n_spots)
    for y0, x0, amp in zip(ys, xs, amps):
        frame += _gaussian_spot(shape, y0, x0, SIGMA_PSF, amp)
    frame += np.random.normal(0, NOISE_STD, shape)
    return np.random.poisson(np.clip(frame, 0, None)).astype(np.float32)


def _localize(frame: np.ndarray, rel_threshold: float, roi_size: int, psf_sigma: float = 1.5):
    """Return localization binary image and parameter array using MicroEye (if available)."""
    
    _, params, *_ = localize_frame(
        0,
        frame,
        frame,
        None,
        _pre_filter,
        _peak_detector,
        rel_threshold,
        np.array([psf_sigma]),
        roi_size,
        FittingMethod._2D_Phasor_CPU,
    )

    loc_im = np.zeros_like(frame, dtype=np.uint16)
    if params is not None and params.size:
        xs = params[:, 0].astype(int)
        ys = params[:, 1].astype(int)
        loc_im[ys, xs] = 1
    return loc_im, params


# ─────────────────────────── Arkitekt endpoints ───────────────────────

@register
def storm_frames(n_frames: int = N_FRAMES):
    """Stream simulated STORM frames as Mikro images via Arkitekt."""
    for i in range(n_frames):
        frame = _simulate_frame(FRAME_SHAPE, np.random.randint(*SPOTS_PER_FRAME))
        loc_im, _ = _localize(frame, REL_THRESHOLD, ROI_SIZE, SIGMA_PSF)

        progress(int((i + 1) / n_frames * 100), f"streaming {i + 1}/{n_frames}")

        yield from_array_like(
            xr.DataArray(loc_im.astype(np.uint16), dims=list("yx")),
            name=f"storm_render_{i:04d}",
        )


@register
def reconstruct_storm_frames(
    images: List[Image],
    threshold: float = 0.4,
    roi_size: int = 13,
    psf_sigma: float = 1.5,
    accumulate: bool = True,
    name: str = "storm_reconstruction",
) -> Image:
    """Reconstruct STORM localization map from Mikro images."""

    recon: np.ndarray | None = None
    stage = create_stage(name="STORM Recon Stage")

    total = len(images)
    for i, img in enumerate(images):
        data = img.data.compute()

        # Flatten to 2‑D grayscale
        if data.ndim == 3:
            sel_dim = 0 if "c" not in img.data.dims else {
                (d, idx) for d, idx in zip(img.data.dims, range(data.ndim)) if d == "c"
            }.pop()[1]
            data = data[sel_dim]
        data = np.asarray(data, dtype=np.float32)

        loc_im, _ = _localize(data, threshold, roi_size, psf_sigma)

        recon = loc_im if recon is None else (recon + loc_im) if accumulate else loc_im

        progress(int((i + 1) / total * 100), f"processed {i + 1}/{total}")

    if recon is None:
        raise ValueError("No frames supplied for reconstruction")

    affine_view = PartialAffineTransformationViewInput(
        affineMatrix=[
            [1.0, 0, 0, 0],
            [0, 1.0, 0, 0],
            [0, 0, 1.0, 0],
            [0, 0, 0, 1.0],
        ],
        stage=stage,
    )

    rgb_view = dict(
        cMin=0,
        cMax=1,
        contrastLimitMax=float(recon.max()),
        contrastLimitMin=float(recon.min()),
        colorMap=ColorMap.MAGENTA,
        baseColor=[0, 0, 0],
    )

    return from_array_like(
        xr.DataArray(recon.astype(np.float32), dims=list("yx")),
        name=name,
        rgb_views=[rgb_view],
        transformation_views=[affine_view],
    )



def _run_simulation(n_frames: int = N_FRAMES, save: bool = True):
    recon_sum = np.zeros(FRAME_SHAPE, dtype=np.uint32)
    all_params = []

    t0 = time.time()
    for i in range(n_frames):
        frame = _simulate_frame(FRAME_SHAPE, np.random.randint(*SPOTS_PER_FRAME))
        render, params = _localize(frame, REL_THRESHOLD, ROI_SIZE, SIGMA_PSF)
        recon_sum += render.astype(np.uint32)
        if params is not None and params.size:
            all_params.append(params)
    dt = time.time() - t0
    print(f"{n_frames} frames processed in {dt:.2f}s ({n_frames / dt:.1f} fps)")

    if save:
        outdir = Path("storm_sim_output")
        outdir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tif.imwrite(outdir / f"reconstruction_{ts}.tif", recon_sum.astype(np.uint16))
        if all_params:
            np.save(outdir / f"localizations_{ts}.npy", np.vstack(all_params))
        print(f"Results saved to {outdir.resolve()}")



def main():
    print("Starting Arkitekt service …")
    app.enter()
    app.run()  # blocks


if __name__ == "__main__":
    main()

"""
Harbor-scale super-resolution pipeline: prognostic model -> CBottleInfill
(variable completion) -> CBottleSR (regional super-resolution to ~5-10km).

This is the "Option B" pipeline from the original CBottleSR integration plan,
rebuilt on top of earth2studio 0.18.0 with two fixes over the original PoC:

  1. CBottleInfill.load_model() was previously given an arbitrary 3-variable
     slice (`[v for v in fcn_vars if v in [...]][:3]`). It now uses the full
     intersection of what the prognostic model actually outputs and what
     CBottleInfill supports as conditioning - more real signal in, less for
     the diffusion model to have to invent.
  2. When a Met Office SST feed is configured, an "sst" slot is appended to
     the conditioning tensor and filled with the observed value (see
     datasources.overlay_metoffice_sst) instead of leaving SST entirely to
     CBottleInfill's built-in monthly climatology. Sea surface temperature
     is a first-order control on marine fog, boundary-layer stability, and
     how much a low can intensify over the sea - worth getting right for
     North Atlantic / North Sea storm work.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch

import config
import datasources
import regions

logger = logging.getLogger(__name__)

RESOLUTION_GRID = {
    "10km": (2161, 4320),
    "5km": (4321, 8640),
}


class SuperResolutionPipeline:
    """Sequentially-loaded CBottleInfill -> CBottleSR stage, run after the
    main prognostic forecast. Models are loaded on first use and can be
    released with .cleanup() to free GPU memory between requests."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._infill = None
        self._infill_input_vars: List[str] = []
        self._super_res = None
        self._sr_resolution = None
        self._sr_window = None

    def _load_infill(self, input_variables: List[str], sampler_steps: int):
        if self._infill is not None and self._infill_input_vars == input_variables:
            return self._infill

        from earth2studio.models.dx import CBottleInfill

        logger.info(f"Loading CBottleInfill (inputs: {input_variables})")
        self._infill = CBottleInfill.load_model(
            CBottleInfill.load_default_package(),
            input_variables=input_variables,
            sampler_steps=sampler_steps,
        ).to(self.device)
        self._infill_input_vars = input_variables
        return self._infill

    def _load_super_res(self, resolution: str, window: Tuple[float, float, float, float], sampler_steps: int):
        if self._super_res is not None and self._sr_resolution == resolution and self._sr_window == window:
            return self._super_res

        from earth2studio.models.dx import CBottleSR

        output_resolution = RESOLUTION_GRID[resolution]
        logger.info(f"Loading CBottleSR ({resolution}, window={window})")
        self._super_res = CBottleSR.load_model(
            CBottleSR.load_default_package(),
            lat_lon=True,
            output_resolution=output_resolution,
            super_resolution_window=window,
            sampler_steps=sampler_steps,
            seed=42,
        ).to(self.device)
        self._sr_resolution = resolution
        self._sr_window = window
        return self._super_res

    def infill_input_variables(self, available_vars: List[str]) -> List[str]:
        """Largest usable subset of the prognostic model's output that
        CBottleInfill can condition on, plus 'sst' when a real feed is
        configured (appended as an extra slot, not sourced from the
        prognostic model - see _augment_with_sst)."""
        from earth2studio.models.dx import CBottleInfill

        supported = set(str(v) for v in CBottleInfill.output_variables)
        chosen = [v for v in available_vars if v in supported]

        if not chosen:
            # Always-safe fallback: every prognostic model here produces these.
            chosen = [v for v in ["u10m", "v10m", "t2m"] if v in available_vars]

        if config.METOFFICE_SST_URL and "sst" not in chosen:
            chosen = chosen + ["sst"]

        return chosen

    def run(
        self,
        data: torch.Tensor,
        coords: dict,
        region_name: str,
        resolution: str = "10km",
        sampler_steps: int = 18,
        time=None,
    ) -> Tuple[torch.Tensor, dict]:
        """Run the Infill -> SR stage on one prognostic model output.
        `data`/`coords` follow earth2studio's (tensor, CoordSystem) convention.
        Returns the super-resolved (tensor, coords) pair."""
        from earth2studio.utils.coords import map_coords

        area = regions.ALL_AREAS.get(region_name.lower().replace(" ", "_"))
        window = area.bounds if area else (48, -4, 52, 2)

        available_vars = [str(v) for v in coords["variable"]]
        input_vars = self.infill_input_variables(available_vars)

        needs_sst_slot = "sst" in input_vars and "sst" not in available_vars
        if needs_sst_slot:
            data, coords = _append_empty_variable(data, coords, "sst")
            data, coords = datasources.overlay_metoffice_sst(data, coords, time=time)

        infill = self._load_infill(input_vars, sampler_steps)
        infill_input, infill_coords = map_coords(data, coords, infill.input_coords())
        infilled_data, infilled_coords = infill(infill_input, infill_coords)
        logger.info(f"CBottleInfill complete: {infilled_data.shape}")

        super_res = self._load_super_res(resolution, window, sampler_steps)
        sr_input, sr_coords = map_coords(infilled_data, infilled_coords, super_res.input_coords())
        high_res_data, high_res_coords = super_res(sr_input, sr_coords)
        logger.info(f"CBottleSR complete: {high_res_data.shape} (window={window})")

        return high_res_data, high_res_coords

    def cleanup(self):
        self._infill = None
        self._super_res = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _append_empty_variable(data: torch.Tensor, coords: dict, name: str):
    """Append a new variable slot (filled with NaN) onto the variable axis of
    a (tensor, coords) pair, ahead of overlaying real data onto it."""
    var_axis = list(coords.keys()).index("variable")
    pad_shape = list(data.shape)
    pad_shape[var_axis] = 1
    pad = torch.full(pad_shape, float("nan"), device=data.device, dtype=data.dtype)
    new_data = torch.cat([data, pad], dim=var_axis)

    new_coords = dict(coords)
    new_coords["variable"] = np.concatenate([coords["variable"], [name]])
    return new_data, new_coords


_pipeline: Optional[SuperResolutionPipeline] = None


def get_pipeline() -> SuperResolutionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SuperResolutionPipeline()
    return _pipeline

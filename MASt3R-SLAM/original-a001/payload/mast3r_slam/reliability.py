"""A a001: bounded two-view static-consistency frontend prototype.

No GT, extra network forward, persistent feature cache, backend or fusion weights.
Unknown observations retain weight 1; only visible depth-consistent evidence is scored.
"""
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F


DEFAULTS = dict(
    temperature=0.3, min_weight=0.05, depth_eps=1e-6,
    depth_abs_tol=0.05, depth_rel_tol=0.10,
    min_support=64, min_evidence_fraction=0.2, min_high_fraction=0.1,
    high_weight=0.25, min_mean_weight=0.2, min_cells=4,
    max_initial_pixel_error=8.0, max_initial_log_depth_error=0.5,
    max_refine_iters=5, max_condition=1e7,
    max_translation_depth_ratio=0.1, max_rotation=0.15, max_log_scale=0.1,
    max_pixel_error_ratio=1.25, pixel_error_slack=0.5,
    save_every=30, max_diagnostics=8,
)
# Codes are intentionally distinct: invisibility is not evidence of motion.
STATUS = dict(visible=0, invalid=1, nonpositive=2, outside=3, missing_depth=4,
              occluded=5, depth_disagreement=6, invalid_feature=7)


def sample_image(image, uv):
    """HWC, pixel centers at integer (u,v), row-major flatten; no resizing."""
    h, w, _ = image.shape
    grid = torch.stack((2 * (uv[:, 0] + 0.5) / w - 1,
                        2 * (uv[:, 1] + 0.5) / h - 1), dim=-1)
    return F.grid_sample(image.permute(2, 0, 1)[None], grid[None, None],
                         mode="bilinear", padding_mode="zeros",
                         align_corners=False)[0, :, 0].T


@torch.inference_mode()
def estimate(X_in_target, source_desc, target_desc, target_depth, K, cfg,
             target_conf=None, confidence_threshold=0.0):
    """Points and depth are both in the TARGET canonical camera gauge.

    source_desc is already gathered using idx_f2k. target_desc is from the SAME
    asymmetric decoder call, indexed at STATIC projections, never at matches.
    Pair output Xkf (in source camera coordinates) must not be used as target depth.
    """
    h, w, d = target_desc.shape
    if source_desc.shape != (X_in_target.shape[0], d):
        raise ValueError("A descriptor/index shape mismatch")
    if target_depth.shape != (h, w):
        raise ValueError("A target depth resolution mismatch")
    z = X_in_target[:, 2]
    p = X_in_target @ K.T
    uv = p[:, :2] / p[:, 2:3]
    finite = torch.isfinite(X_in_target).all(-1) & torch.isfinite(uv).all(-1)
    positive = z > cfg["depth_eps"]
    inside = (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)
    safe_uv = torch.where((finite & positive & inside)[:, None], uv, 0)
    # Invalid depth/confidence must not be interpolated into valid evidence.
    depth_ok = torch.isfinite(target_depth) & (target_depth > cfg["depth_eps"])
    if target_conf is not None:
        depth_ok &= torch.isfinite(target_conf) & (target_conf > confidence_threshold)
    sampled_ok = sample_image(depth_ok[..., None].float(), safe_uv)[:, 0] > 0.9999
    depth = sample_image(torch.nan_to_num(target_depth)[..., None], safe_uv)[:, 0]
    tol = cfg["depth_abs_tol"] + cfg["depth_rel_tol"] * depth.abs()
    occluded = z > depth + tol
    disagree = z < depth - tol
    feature_ok = torch.isfinite(target_desc).all(-1) & (torch.linalg.vector_norm(torch.nan_to_num(target_desc), dim=-1) > 1e-8)
    sampled_feature_ok = sample_image(feature_ok[..., None].float(), safe_uv)[:, 0] > 0.9999
    sampled = sample_image(torch.nan_to_num(target_desc), safe_uv)
    source_ok = torch.isfinite(source_desc).all(-1) & (torch.linalg.vector_norm(torch.nan_to_num(source_desc), dim=-1) > 1e-8)
    sampled_feature_ok &= torch.linalg.vector_norm(sampled, dim=-1) > 1e-8
    error = (1 - (F.normalize(torch.nan_to_num(source_desc), dim=-1) * F.normalize(sampled, dim=-1)).sum(-1)).clamp(0, 2)
    status = torch.full_like(z, STATUS["visible"], dtype=torch.uint8)
    # Assign from lower to higher priority.
    status[~(source_ok & sampled_feature_ok)] = STATUS["invalid_feature"]
    status[disagree] = STATUS["depth_disagreement"]
    status[occluded] = STATUS["occluded"]
    status[~sampled_ok] = STATUS["missing_depth"]
    status[~inside] = STATUS["outside"]
    status[~positive] = STATUS["nonpositive"]
    status[~finite] = STATUS["invalid"]
    evidence = status == STATUS["visible"]
    weight = torch.ones_like(z)
    weight[evidence] = torch.exp(-error[evidence] / cfg["temperature"]).clamp(cfg["min_weight"], 1)
    return dict(weight=weight[:, None], evidence=evidence, status=status,
                uv=uv, error=torch.where(evidence, error, float("nan")))


def robust_cost(whitened_r, weight, k):
    """Original tracker uses componentwise Huber, preserved here: sum w rho(S0*r)."""
    a = whitened_r.abs()
    rho = torch.where(a <= k, 0.5 * a.square(), k * (a - 0.5 * k))
    return float((weight * rho).sum())


def check_normal_matrix(H, limit):
    if not torch.isfinite(H).all() or (H.diag() <= 0).any():
        raise ValueError("nonfinite_or_degenerate_normal_matrix")
    scale = H.diag().sqrt()
    eig = torch.linalg.eigvalsh(H / (scale[:, None] * scale[None]))
    if eig[0] <= 0 or float(eig[-1] / eig[0]) > limit:
        raise ValueError("ill_conditioned_normal_matrix")


class FrontendReliability:
    def __init__(self, options, tracking_cfg):
        self.mode = options["mode"]
        self.cfg = dict(DEFAULTS)
        unknown = set(options) - set(self.cfg) - {"mode", "diagnostic_dir"}
        if unknown:
            raise ValueError(f"Unknown A options: {sorted(unknown)}")
        self.cfg.update({k: v for k, v in options.items() if k in self.cfg})
        c = self.cfg
        for key, value in c.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid A option {key}")
        for key in ("min_support", "min_cells", "max_refine_iters", "save_every", "max_diagnostics"):
            if int(c[key]) != c[key]:
                raise ValueError(f"A {key} must be an integer")
        if not (0 < c["temperature"] and 0 < c["min_weight"] <= 1 and
                1 <= c["max_refine_iters"] <= 10 and c["min_support"] >= 8 and
                1 <= c["min_cells"] <= 9 and c["max_condition"] > 1 and
                0 < c["high_weight"] <= 1):
            raise ValueError("A options outside supported bounds")
        for key in ("min_evidence_fraction", "min_high_fraction", "min_mean_weight"):
            if not 0 <= c[key] <= 1:
                raise ValueError(f"Invalid A fraction {key}")
        self.tracking_cfg = tracking_cfg
        self.directory = Path(os.environ.get("MAST3R_A_DIAGNOSTICS", options.get("diagnostic_dir", "logs/a_diagnostics")))
        self.calls = self.saved = self.attempts = self.accepted = self.fallbacks = 0
        print("A_CONFIG " + json.dumps(dict(mode=self.mode, scope="two_view_frontend", options=c)), flush=True)

    def record(self, frame_id, reason, **stats):
        if reason not in ("accepted", "diagnostic"):
            self.fallbacks += 1
        print("A_EVENT " + json.dumps(dict(frame=int(frame_id), mode=self.mode, reason=reason,
              attempts=self.attempts, accepted=self.accepted, fallbacks=self.fallbacks,
              **stats), allow_nan=False), flush=True)

    def initial_failure(self, frame_id, reason):
        self.record(frame_id, reason, a_ms=0.0)

    def _residual(self, tracker, pose, Xf, meas, K, size, active):
        from mast3r_slam.geometry import project_calib
        pz, valid = project_calib(pose.act(Xf), K, size, border=tracker.cfg["pixel_border"], z_eps=tracker.cfg["depth_eps"])
        if not torch.isfinite(pz[active]).all() or not valid[active].all():
            raise ValueError("lost_projection_support")
        return meas - pz, valid

    def _support(self, result, valid, size):
        c = self.cfg
        active = valid[:, 0]
        evidence = result["evidence"] & active
        n = int(active.sum()); ne = int(evidence.sum())
        high = evidence & (result["weight"][:, 0] >= c["high_weight"])
        weights = result["weight"][active]
        if n == 0 or ne < c["min_support"] or ne / n < c["min_evidence_fraction"]:
            raise ValueError("insufficient_visible_evidence")
        if int(high.sum()) < c["min_support"] or float(high.sum()) / n < c["min_high_fraction"]:
            raise ValueError("all_low_or_insufficient_static_support")
        if float(weights.mean()) < c["min_mean_weight"] or float(weights.sum().square() / weights.square().sum()) < c["min_support"]:
            raise ValueError("insufficient_effective_support")
        h, w = size
        uv = result["uv"][high]
        cell = (uv[:, 0] * 3 / w).long().clamp(0, 2) + 3 * (uv[:, 1] * 3 / h).long().clamp(0, 2)
        if int(torch.unique(cell).numel()) < c["min_cells"]:
            raise ValueError("insufficient_spatial_support")
        return ne

    def _save(self, frame_id, result, size):
        c = self.cfg
        if not c["save_every"] or self.calls % c["save_every"] != 0 or self.saved >= c["max_diagnostics"]:
            return
        import numpy as np
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = self.directory / f"frame_{int(frame_id):06d}"
        arrays = {k: v.detach().cpu().numpy() for k, v in result.items()}
        # Exclusive creation prevents a manual run from overwriting old diagnostics.
        with stem.with_suffix(".npz").open("xb") as f:
            np.savez_compressed(f, **arrays, image_size=np.array(size))
        h, w = size
        status = arrays["status"].reshape(h, w)
        uv = np.nan_to_num(arrays["uv"]).reshape(h, w, 2)
        error = np.nan_to_num(arrays["error"], nan=0).reshape(h, w)
        weight = arrays["weight"].reshape(h, w)
        validity = np.zeros((h, w, 3)); validity[..., 1] = status == 0; validity[..., 0] = status / 7
        projection = np.zeros((h, w, 3)); projection[..., :2] = uv / [max(w-1, 1), max(h-1, 1)]
        errors = np.stack((error / 2, 1 - error / 2, np.zeros_like(error)), -1)
        weights = np.stack((1 - weight, weight, np.zeros_like(weight)), -1)
        panels = np.uint8(np.clip(np.concatenate((validity, projection, errors, weights), axis=1), 0, 1) * 255)
        with stem.with_suffix(".ppm").open("xb") as f:
            f.write(f"P6\n{w*4} {h}\n255\n".encode()); f.write(panels.tobytes())
        self.saved += 1

    @torch.inference_mode()
    def run(self, tracker, frame_id, descriptors, idx_f2k, Xf, Xk, Ck,
            initial_world, key_world, initial_relative, Qk, valid, meas, valid_meas,
            K, size, initial_status, target_frame_id=None):
        device = Xf.device
        if device.type == "cuda": torch.cuda.synchronize(device)
        start = time.perf_counter(); self.calls += 1
        reason = "diagnostic"; stats = {"target_frame": target_frame_id}; output = (initial_world, initial_relative)
        try:
            if not initial_status.get("converged", False):
                raise ValueError("initial_not_converged")
            if not torch.isfinite(initial_relative.matrix()).all():
                raise ValueError("nonfinite_initial_pose")
            from mast3r_slam.geometry import project_calib
            initial_points = initial_relative.act(Xf)
            pz, projected = project_calib(initial_points, K, size, border=tracker.cfg["pixel_border"], z_eps=tracker.cfg["depth_eps"])
            active = (valid & valid_meas & projected)[:, 0]
            if int(active.sum()) < self.cfg["min_support"] or float(active.sum()) / max(int(valid.sum()), 1) < 0.5:
                raise ValueError("insufficient_initial_projection_support")
            r0 = meas - pz
            if not torch.isfinite(r0[active]).all(): raise ValueError("nonfinite_initial_residual")
            pixel0 = float(r0[active, :2].norm(dim=-1).median())
            if pixel0 > self.cfg["max_initial_pixel_error"] or float(r0[active, 2].abs().median()) > self.cfg["max_initial_log_depth_error"]:
                raise ValueError("unreliable_initial_pose")
            source_desc, target_desc = descriptors
            h, w = size
            if source_desc.shape[:2] != (h, w) or target_desc.shape[:2] != (h, w) or Xk.shape[0] != h*w:
                raise ValueError("unsupported_feature_resolution")
            result = estimate(initial_points, source_desc.reshape(h*w, -1)[idx_f2k], target_desc,
                              Xk[:, 2].reshape(h, w), K, self.cfg, Ck[:, 0].reshape(h, w), tracker.cfg["C_conf"])
            result.update(idx_f2k=idx_f2k, K=K, T_CkCf=initial_relative.matrix())
            weights = result["weight"].detach()
            vals = weights[valid[:, 0]]
            stats.update(support=int(active.sum()), evidence=int((result["evidence"] & valid[:, 0]).sum()),
                         weight_mean=float(vals.mean()), weight_min=float(vals.min()),
                         weight_quantiles=[float(x) for x in torch.quantile(vals, torch.tensor([.1,.5,.9], device=device))],
                         states={k: int((result["status"] == v).sum()) for k,v in STATUS.items()}, initial_pixel_median=pixel0)
            self._save(frame_id, result, size)
            self._support(result, valid & valid_meas & projected, size)
            if self.mode == "weighted":
                self.attempts += 1
                refinement = dict(max_iters=int(self.cfg["max_refine_iters"]), max_condition=self.cfg["max_condition"])
                world, relative = tracker.opt_pose_calib_sim3(Xf, Xk, initial_world, key_world, Qk, valid,
                     meas, valid_meas, K, size, weights=weights, refinement=refinement)
                stats["refine_steps"] = refinement.get("steps", 0)
                if not refinement.get("converged", False): raise ValueError("refine_not_converged")
                r1, projected1 = self._residual(tracker, relative, Xf, meas, K, size, active)
                if int((projected1 & valid & valid_meas).sum()) < 0.98 * int(active.sum()): raise ValueError("lost_projection_support")
                delta = (relative * initial_relative.inv()).log()
                if not torch.isfinite(delta).all(): raise ValueError("nonfinite_pose_update")
                depth = float(initial_points[active, 2].median())
                if (float(delta[..., :3].norm()) / max(depth, 1e-6) > self.cfg["max_translation_depth_ratio"] or
                    float(delta[..., 3:6].norm()) > self.cfg["max_rotation"] or float(delta[..., 6:].abs().max()) > self.cfg["max_log_scale"]):
                    raise ValueError("excessive_pose_update")
                pixel1 = float(r1[active, :2].norm(dim=-1).median())
                if pixel1 > pixel0 * self.cfg["max_pixel_error_ratio"] + self.cfg["pixel_error_slack"]:
                    raise ValueError("unweighted_pixel_regression")
                sigma = Xf.new_tensor([tracker.cfg["sigma_pixel"], tracker.cfg["sigma_pixel"], tracker.cfg["sigma_depth"]])
                s0 = torch.sqrt(Qk[active]) / sigma
                cost0 = robust_cost(s0*r0[active], weights[active], tracker.cfg["huber"])
                cost1 = robust_cost(s0*r1[active], weights[active], tracker.cfg["huber"])
                if not math.isfinite(cost1) or cost1 > cost0 + max(1e-6, cost0*1e-5): raise ValueError("robust_cost_regression")
                stats.update(final_pixel_median=pixel1, cost_initial=cost0, cost_final=cost1)
                output = (world, relative); self.accepted += 1; reason = "accepted"
        except (ValueError, RuntimeError, OSError) as error:
            reason = str(error).splitlines()[0][:200]
        if device.type == "cuda": torch.cuda.synchronize(device)
        self.record(frame_id, reason, a_ms=(time.perf_counter()-start)*1000, **stats)
        return output

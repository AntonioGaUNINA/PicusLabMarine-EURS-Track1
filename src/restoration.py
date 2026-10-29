"""Portable #19386 restoration from one degraded PNG and suffix indices a,b.

The local equations preserve R113's R26 prior, with the packaged Full20
DRUNet, and its selected canonical depth posterior. No scene inventory,
reference image, label, cached prediction, or scientific controller is used.
CUDA inference must run in FP32 with TF32 disabled by the calling process.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

if __package__:
    from .components import load_component
    from .image_io import decode
else:
    from components import load_component
    from image_io import decode


ROUNDING_BAND = 1e-6


def _read(path):
    return json.loads(Path(path).read_text())


def _verified_field(directory, grid):
    directory = Path(directory)
    contract = _read(directory / "model_contract.json")
    assert contract["selected_grid"] == grid
    with (directory / "weights.npz").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == contract["weights_sha256"]
    return contract


def _condition(stem):
    a, b = map(int, stem.rsplit("_", 2)[1:])
    assert 1 <= a <= 10 and b in range(5, 101, 5)
    return (a - 1) * 20 + b // 5 - 1


def _key(stem):
    return "_".join(str(int(x)) for x in stem.rsplit("_", 2)[1:])


class QuadraticField:
    """R11 quadratic FP32 field, including its FP64 constant fallback."""

    def __init__(self, directory):
        contract = _verified_field(directory, 8)
        assert contract["status"] == "PASS"
        with np.load(Path(directory) / "weights.npz", allow_pickle=False) as archive:
            self.bank = {key.removeprefix("g8_"): archive[key].copy()
                         for key in archive.files if key.startswith("g8_")}
            self.fallback = archive["fallback64"].copy()

    def floating(self, values, stem):
        # native_refinement.floating; the inverse remains unclipped here.
        coeff = self.bank.get(_key(stem))
        h, w = values.shape[:2]
        if coeff is None:
            return np.broadcast_to(self.fallback, (h, w, 3))
        r, g, b = np.moveaxis(values.astype(np.float32) / 255, -1, 0)
        features = np.stack([r, g, b, r*r, g*g, b*b, r*g, r*b, g*b,
                             np.ones_like(r)], -1)
        maps = F.interpolate(
            torch.from_numpy(coeff.reshape(8, 8, 30)).permute(2, 0, 1)[None],
            size=(h, w), mode="bilinear", align_corners=False,
        )[0].permute(1, 2, 0).numpy().reshape(h, w, 10, 3)
        return (features[..., None] * maps).sum(-2)

    def predict(self, values, stem):
        return (np.clip(self.floating(values, stem), 0, 1) * 255).astype(np.uint8)


class GridRestorer:
    """R06 affine field and local quantization-noise propagation."""

    def __init__(self, directory):
        contract = _verified_field(directory, 4)
        with np.load(Path(directory) / "weights.npz", allow_pickle=False) as archive:
            self.bank = {name.split("_", 1)[1]: archive[name].copy()
                         for name in archive.files if name.startswith("g4_")}
        self.fallback = np.array(contract["fallback_float64"], np.float64)
        assert len(self.bank) == 198

    def predict(self, rgb, stem, with_sigma=False):
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
        coeff = self.bank.get(_key(stem))
        h, w = rgb.shape[:2]
        if coeff is None:
            prediction = np.broadcast_to(self.fallback, (h, w, 3))
            sigma = np.zeros((h, w), np.float32) if with_sigma else None
        else:
            field = F.interpolate(
                torch.from_numpy(coeff.reshape(4, 4, 12)).permute(2, 0, 1)[None],
                size=(h, w), mode="bilinear", align_corners=False,
            )[0].permute(1, 2, 0).numpy().reshape(h, w, 4, 3)
            values = rgb.astype(np.float32) / 255
            prediction = (values[..., None] * field[..., :3, :]).sum(-2) + field[..., 3, :]
            sigma = None
            if with_sigma:
                active = (prediction > 0) & (prediction < 1)
                variance = (field[..., :3, :] ** 2).sum(-2) / (12 * 255**2)
                sigma = np.sqrt((variance * active).mean(-1)).astype(np.float32)
        output = (np.clip(prediction, 0, 1) * 255).astype(np.uint8)
        return (output, sigma) if with_sigma else output


class Prior:
    """R08 DRUNet inference, with the packaged Full20 physical025 weights."""

    def __init__(self, model_dir, device):
        self.device = torch.device(device)
        self.model = load_component("drunet", root=model_dir, device=self.device)

    @torch.inference_mode()
    def __call__(self, rgb, sigma, scale=1.):
        if scale == 0 or sigma.max() == 0:
            return rgb.copy()
        h, w = rgb.shape[:2]
        values = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(self.device).float() / 255
        noise = torch.from_numpy(np.clip(sigma * scale, 0, 50 / 255))[None, None].to(self.device)
        inputs = torch.cat([values, noise], 1)
        inputs = F.pad(inputs, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
        prediction = self.model(inputs)[..., :h, :w]
        assert torch.isfinite(prediction).all()
        return (prediction.clamp(0, 1) * 255).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()


class Photo:
    """infer_eurs.Photo: R06 -> DRUNet, averaged with quantized R11."""

    def __init__(self, model_dir, device):
        self.field = QuadraticField(Path(model_dir) / "coefficients/r11")
        self.grid = GridRestorer(Path(model_dir) / "coefficients/r06")
        self.prior = Prior(model_dir, device)

    def quadratic(self, values, stem):
        return self.field.predict(values, stem)

    def __call__(self, sample):
        r12 = self.quadratic(sample["values"], sample["stem"])
        r6, sigma = self.grid.predict(sample["rgb"], sample["stem"], True)
        r8 = self.prior(r6, sigma, 1.)
        r13 = np.clip(r8.astype(np.float32) + np.float32(.5) *
                      (r12.astype(np.float32) - r8.astype(np.float32)), 0, 255).astype(np.uint8)
        return r13, r12


def normalize_depth(depth):
    """depth_prior.normalize, preserving the model's FP32 arithmetic."""
    low, high = float(depth.min()), float(depth.max())
    return ((depth - low) / (high - low)).astype(np.float32) if high - low > 1e-12 else np.zeros_like(depth, dtype=np.float32)


def sample_depth(normalized, source_size, ys, xs):
    """depth_prior.sample: CPU FP32 center sampling, align_corners=False."""
    width, height = source_size
    x, y = np.meshgrid(2 * (np.asarray(xs, dtype=float) + .5) / width - 1,
                       2 * (np.asarray(ys, dtype=float) + .5) / height - 1)
    grid = torch.from_numpy(np.stack([x, y], -1).astype(np.float32))[None]
    values = torch.from_numpy(np.asarray(normalized, dtype=np.float32))[None, None]
    return F.grid_sample(values, grid, mode="bilinear", padding_mode="border",
                         align_corners=False)[0, 0].numpy().copy()


def numpy_forward(x, xy, q, weight):
    """Exact nonlinear_consistency.forward + depth_consistency.forward."""
    base = np.vstack([weight[:20], weight[25]])
    r, g, b = x.T
    u, v = xy.T
    features = np.column_stack([r, g, b, r*r, g*g, b*b, r*g, r*b, g*b,
                                u, v, u*u, v*v, u*v, r*u, g*u, b*u,
                                r*v, g*v, b*v, np.ones(len(x))])
    derivative = np.zeros((len(x), 21, 3))
    derivative[:, 0, 0] = 1
    derivative[:, 1, 1] = 1
    derivative[:, 2, 2] = 1
    derivative[:, 3, 0] = 2*r
    derivative[:, 4, 1] = 2*g
    derivative[:, 5, 2] = 2*b
    derivative[:, 6, 0] = g
    derivative[:, 6, 1] = r
    derivative[:, 7, 0] = b
    derivative[:, 7, 2] = r
    derivative[:, 8, 1] = b
    derivative[:, 8, 2] = g
    for i in range(3):
        derivative[:, 14+i, i] = u
        derivative[:, 17+i, i] = v
    value = features @ base / 255
    jac = np.einsum("nki,kj->nji", derivative, base) / 255
    # q is FP32 from sample_depth. q*q MUST round before promotion into extra.
    extra = np.column_stack([q, q*q, x*q[:, None]])
    value += extra @ weight[20:25] / 255
    jac += q[:, None, None] * weight[22:25].T[None] / 255
    return value, jac


def numpy_project(base, z, xy, q, weight, vp, vo):
    """Reference posterior, also used to repair ambiguous CUDA rounding."""
    prior = base.astype(float) / 255
    x = prior.copy()
    valid = (z > 0) & (z < 1)
    costs = []

    def cost(values):
        prediction, _ = numpy_forward(values, xy, q, weight)
        return np.square(values-prior).sum(1) / vp + np.square((prediction-z)*valid).sum(1) / vo

    for _ in range(3):
        prediction, jac = numpy_forward(x, xy, q, weight)
        jac *= valid[:, :, None]
        h = np.einsum("nji,njk->nik", jac, jac) / vo + np.eye(3)[None] / vp
        gradient = (x-prior) / vp + np.einsum("nji,nj->ni", jac, (prediction-z)*valid) / vo
        step = np.linalg.solve(h, -gradient[:, :, None])[:, :, 0]
        before = cost(x)
        best = before.copy()
        chosen = x.copy()
        for alpha in (1., .5, .25, .125):
            trial = np.clip(x + alpha*step, 0, 1)
            value = cost(trial)
            keep = value < best
            best[keep] = value[keep]
            chosen[keep] = trial[keep]
        assert np.all(best <= before + 1e-10)
        x = chosen
        costs.append(float(best.mean()))
    assert np.isfinite(x).all() and all(b <= a + 1e-10 for a, b in zip(costs, costs[1:]))
    out = np.rint(np.clip(x * 255, 0, 255)).astype(np.uint8)
    assert np.array_equal(out[~valid.any(1)], base[~valid.any(1)])
    return out, valid, costs


def torch_forward(x, xy, q, weight):
    """cuda_depth_projection.forward with PRESERVE_Q_PRECISION=True."""
    r, g, b = x.unbind(1)
    u, v = xy.unbind(1)
    features = torch.stack([r, g, b, r*r, g*g, b*b, r*g, r*b, g*b, u, v,
                            u*u, v*v, u*v, r*u, g*u, b*u, r*v, g*v, b*v,
                            q, q*q, r*q, g*q, b*q, torch.ones_like(r)], 1)
    derivative = torch.zeros((len(x), 26, 3), dtype=x.dtype, device=x.device)
    derivative[:, 0, 0] = 1
    derivative[:, 1, 1] = 1
    derivative[:, 2, 2] = 1
    derivative[:, 3, 0] = 2*r
    derivative[:, 4, 1] = 2*g
    derivative[:, 5, 2] = 2*b
    derivative[:, 6, 0] = g
    derivative[:, 6, 1] = r
    derivative[:, 7, 0] = b
    derivative[:, 7, 2] = r
    derivative[:, 8, 1] = b
    derivative[:, 8, 2] = g
    for i in range(3):
        derivative[:, 14+i, i] = u
        derivative[:, 17+i, i] = v
        derivative[:, 22+i, i] = q
    features[:, 21] = q.float().square().to(torch.float64)
    base = torch.cat([weight[:20], weight[25:]], dim=0)
    first = torch.cat([features[:, :20], features[:, 25:]], dim=1)
    derivative0 = torch.cat([derivative[:, :20], derivative[:, 25:]], dim=1)
    value = first @ base / 255
    value = value + features[:, 20:25] @ weight[20:25] / 255
    jac = torch.einsum("nki,kj->nji", derivative0, base) / 255
    jac = jac + q[:, None, None] * weight[22:25].T[None] / 255
    return value, jac


def torch_project(base, z, xy, q, weight, vp, vo, prequant=False):
    """Three FP64 Gauss--Newton steps and ordered strict line search."""
    prior = base.to(torch.float64) / 255
    x = prior.clone()
    valid = (z > 0) & (z < 1)

    def cost(value):
        prediction, _ = torch_forward(value, xy, q, weight)
        return (value-prior).square().sum(1) / vp + ((prediction-z)*valid).square().sum(1) / vo

    for _ in range(3):
        pred, jac = torch_forward(x, xy, q, weight)
        jac = jac * valid[:, :, None]
        h = torch.einsum("nji,njk->nik", jac, jac) / vo + torch.eye(3, device=x.device, dtype=x.dtype)[None] / vp
        grad = (x-prior) / vp + torch.einsum("nji,nj->ni", jac, (pred-z)*valid) / vo
        step = torch.linalg.solve(h, -grad[:, :, None])[:, :, 0]
        best = cost(x)
        chosen = x.clone()
        for alpha in (1., .5, .25, .125):
            trial = (x + alpha*step).clamp(0, 1)
            value = cost(trial)
            keep = value < best
            best = torch.where(keep, value, best)
            chosen = torch.where(keep[:, None], trial, chosen)
        x = chosen
    if prequant:
        return (x * 255).clamp(0, 255)
    return (x * 255).clamp(0, 255).round().to(torch.uint8)


@torch.inference_mode()
def project_image(base, observed, q, weight, params, device="cuda", repair_stats=None):
    """R95: 64-row CUDA blocks and up to eight exact NumPy repair workers.

    CPU execution uses the same NumPy reference on every block. Sampling q
    remains CPU FP32 on both devices, exactly as in the historical path.
    """
    if q is None:
        return base.copy()
    device = torch.device(device)
    h, w = base.shape[:2]
    out = base.copy()
    tw = torch.as_tensor(weight, device=device, dtype=torch.float64) if device.type != "cpu" else None
    pending = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for top in range(0, h, 64):
            ys, xs = np.arange(top, min(top+64, h)), np.arange(w)
            u, v = np.meshgrid(2*xs/max(1, w-1)-1, 2*ys/max(1, h-1)-1)
            xy = np.column_stack([u.ravel(), v.ravel()])
            qv = sample_depth(q, (w, h), ys, xs).ravel()
            a = base[top:top+len(ys)].reshape(-1, 3)
            z = observed[top:top+len(ys)].reshape(-1, 3).astype(float) / 255
            if repair_stats is not None:
                repair_stats["blocks"] += 1
            if device.type == "cpu":
                result = numpy_project(a, z, xy, qv, weight, params["vp"], params["vo"])[0]
                out[top:top+len(ys)] = result.reshape(len(ys), w, 3)
                continue
            tensors = [torch.as_tensor(x, device=device, dtype=torch.float64) for x in [a, z, xy, qv]]
            raw = torch_project(*tensors, tw, params["vp"], params["vo"], prequant=True)
            if bool((torch.abs(torch.frac(raw)-.5) <= ROUNDING_BAND).any()):
                if repair_stats is not None:
                    repair_stats["CPU_blocks"] += 1
                pending.append((top, len(ys), pool.submit(numpy_project, a, z, xy, qv,
                                                        weight, params["vp"], params["vo"])))
            else:
                out[top:top+len(ys)] = raw.round().to(torch.uint8).reshape(len(ys), w, 3).cpu().numpy()
        for top, count, future in pending:
            out[top:top+count] = future.result()[0].reshape(count, w, 3)
    return out


class Restoration:
    """Exact #19386 restoration. Model assets are resolved under model_dir."""

    def __init__(self, model_dir, device="cuda"):
        from transformers import AutoImageProcessor

        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.rule = dict(name="a4_b100_mix100", a_max=4, b_max=100, alpha=1.)
        self.photo = Photo(self.model_dir, self.device)
        self.field = self.photo.field
        self.native = load_component("restormer_native", root=self.model_dir, device=self.device)
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_dir / "assets/depth_anything", local_files_only=True,
            trust_remote_code=False, use_fast=False,
        )
        self.depth = load_component("depth_anything_v2_small", root=self.model_dir, device=self.device)
        posterior = self.model_dir / "coefficients/posterior"
        self.selected = set(_read(posterior / "result.json")["selected_conditions"])
        assert len(self.selected) == 87
        with np.load(posterior / "renderer.npz", allow_pickle=False) as archive:
            self.bank = {f"{a}_{b}": weight.copy() for (a, b), weight in
                         zip(archive["conditions"], archive["weights"])}
        self.noise = _read(posterior / "noise.json")
        assert self.selected <= self.bank.keys() and self.selected <= self.noise.keys()
        assert all(np.isfinite(self.bank[key]).all() and self.bank[key].shape == (26, 3)
                   and np.isfinite(self.noise[key]["vp"]) and self.noise[key]["vp"] > 0
                   and np.isfinite(self.noise[key]["vo"]) and self.noise[key]["vo"] > 0
                   for key in self.selected)
        self.repair_stats = {"blocks": 0, "CPU_blocks": 0}

    @torch.inference_mode()
    def restore(self, sample):
        """OwnImagePipeline.restore: R21 for a<=4, otherwise R13."""
        a, b = map(int, sample["stem"].rsplit("_", 2)[1:])
        if a > self.rule["a_max"] or b > self.rule["b_max"]:
            return self.photo(sample)[0]
        inverse = self.field.floating(sample["values"], sample["stem"])
        base = (np.clip(inverse, 0, 1) * 255).astype(np.uint8)
        index = _condition(sample["stem"])
        if sample["nativezero"] or self.native.gates[index].item() == 0:
            return base
        packed = np.concatenate([base.astype(np.float32), sample["values"], inverse.astype(np.float32)*255], -1)
        rgb = torch.from_numpy(packed).permute(2, 0, 1)[None].to(self.device).float() / 255
        h, w = base.shape[:2]
        padded = F.pad(rgb, (0, (-w) % 8, 0, (-h) % 8), mode="replicate")
        refined = self.native(padded, torch.tensor([index], device=self.device))[..., :h, :w]
        values = torch.from_numpy(base.copy()).permute(2, 0, 1).to(self.device).float() + 255*(refined-rgb[:, :3])[0]
        return values.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    @torch.inference_mode()
    def __call__(self, path):
        sample = decode(path)
        prior = self.restore(sample)
        # R113 selects posterior conditions by the literal two suffix tokens;
        # R06/R11 separately canonicalize those tokens inside their fields.
        key = "_".join(sample["stem"].rsplit("_", 2)[1:])
        q = None
        if key in self.selected and not sample["nativezero"]:
            values = self.processor(images=Image.fromarray(prior), return_tensors="pt")
            raw = self.depth(**{k: v.to(self.device) for k, v in values.items()}).predicted_depth[0].cpu().numpy()
            assert np.isfinite(raw).all() and (raw >= 0).all()
            q = normalize_depth(raw)
        output = project_image(prior, sample["values"], q, self.bank[key],
                               self.noise[key], self.device, self.repair_stats) if q is not None else prior.copy()
        assert output.dtype == np.uint8 and output.shape == sample["rgb"].shape
        return output

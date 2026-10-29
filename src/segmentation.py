"""Local S79 plus Full20 classifier reproduction for submission #19433.

The semantic branch uses its own R15 view, independently of R113 restoration.
CUDA/component-output equivalence and full cold memory remain unverified; the
CPU checks exercise extracted arithmetic and transport, not the trained models.
"""
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from .components import load_component
from .image_io import decode
from .sam_worker import frame_read, frame_write


class R15Field:
    """Exact final_photometry.Field inference with model-relative coefficients."""

    def __init__(self, model_dir):
        directory = Path(model_dir) / 'coefficients/r15'
        contract = json.loads((directory / 'model_contract.json').read_text())
        assert contract['status'] == 'PASS' and contract['selected_grid'] == 8
        with (directory / 'weights.npz').open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == contract['weights_sha256']
        with np.load(directory / 'weights.npz', allow_pickle=False) as archive:
            self.bank = {key.removeprefix('g8_'): archive[key].copy()
                         for key in archive.files if key.startswith('g8_')}
            self.fallback = archive['fallback64'].copy()

    def predict(self, values, stem):
        a, b = map(int, stem.rsplit('_', 2)[1:])
        assert a in range(1, 11) and b in range(5, 101, 5)
        coeff = self.bank.get(f'{a}_{b}')
        h, w = values.shape[:2]
        assert values.shape == (h, w, 3) and np.isfinite(values).all()
        if coeff is None:
            output = np.broadcast_to(self.fallback, (h, w, 3))
        else:
            r, g, b = np.moveaxis(values.astype(np.float32) / 255, -1, 0)
            features = np.stack([r, g, b, r*r, g*g, b*b, r*g, r*b, g*b,
                                 np.ones_like(r)], -1)
            field = F.interpolate(
                torch.from_numpy(coeff.reshape(8, 8, 30)).permute(2, 0, 1)[None],
                size=(h, w), mode='bilinear', align_corners=False
            )[0].permute(1, 2, 0).numpy().reshape(h, w, 10, 3)
            output = (features[..., None] * field).sum(-2)
        return (np.clip(output, 0, 1) * 255).astype(np.uint8)


def normalize(rgb, device):
    """advanced.normalize: ImageNet float32 normalization, then zero pad32."""
    x = rgb.to(device, non_blocking=True).float() / 255
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    x = (x - mean) / std
    h, w = x.shape[-2:]
    return F.pad(x, (0, (-w) % 32, 0, (-h) % 32), value=0.)


def _autocast(device):
    # The historical trained-model forward is CUDA BF16. CPU is provided only
    # for arithmetic fixtures; it does not establish CUDA equivalence.
    return (torch.autocast('cuda', dtype=torch.bfloat16)
            if torch.device(device).type == 'cuda' else nullcontext())


def dino_probabilities(model, rgb, device):
    """expanded.s_probabilities, including interpolation BEFORE softmax."""
    h, w = rgb.shape[-2:]
    with _autocast(device):
        logits, _ = model(normalize(rgb, device))
    logits = F.interpolate(
        logits.float(), size=(math.ceil(h / 32) * 32, math.ceil(w / 32) * 32),
        mode='bilinear', align_corners=False
    )[..., :h, :w]
    assert torch.isfinite(logits).all()
    return logits.softmax(1)


def m2f_scores(output, shape):
    """advanced.semantic: query probabilities -> normalized five-class scores."""
    cls = output.class_queries_logits.float().softmax(-1)[..., :5]
    h, w = shape
    padded = (math.ceil(h / 32) * 32, math.ceil(w / 32) * 32)
    masks = F.interpolate(output.masks_queries_logits.float(), size=padded,
                          mode='bilinear', align_corners=False).sigmoid()
    masks = masks[..., :h, :w]
    score = torch.einsum('bqc,bqhw->bchw', cls, masks)
    return score / score.sum(1, keepdim=True).clamp_min(1e-8)


def compose(first, animals, background, support, boxes, black=False):
    """S72/S79 hybrid scores, protected Other, then original S51 local edits.

    The hybrid vector deliberately is NOT normalized before its argmax.
    Returns both the original protected control and the S79 specialist mask.
    """
    assert first.shape == animals.shape == background.shape and first.shape[0] == 5
    assert np.isfinite(first).all() and np.isfinite(animals).all() and np.isfinite(background).all()
    assert np.allclose(first.sum(0), 1, atol=1e-5)
    assert np.allclose(animals.sum(0), 1, atol=1e-5)
    assert support.shape == boxes.shape == first.shape[1:]
    assert support.dtype == boxes.dtype == bool
    baseline = first.argmax(0).astype(np.uint8)
    control_scores = first * np.float32(.7) + animals * np.float32(.3)
    hybrid = animals.copy()
    hybrid[:2] = background[:2]
    candidate_scores = first * np.float32(.7) + hybrid * np.float32(.3)
    assert np.array_equal(control_scores[2:], candidate_scores[2:])
    if black:
        local = baseline.copy()
    else:
        local_scores = first.copy()
        local_scores[4] *= np.where(support, 2., np.where(boxes, .5, 1.)).astype(np.float32)
        local = local_scores.argmax(0).astype(np.uint8)
    result = {}
    for name, mask in [('control', control_scores.argmax(0).astype(np.uint8)),
                       ('specialist', candidate_scores.argmax(0).astype(np.uint8))]:
        protected = np.where((baseline == 4) | (mask == 4), baseline, mask).astype(np.uint8)
        result[name] = np.where(local != baseline, local, protected).astype(np.uint8)
    if black:
        result['specialist'] = result['control'].copy()
    assert np.array_equal(result['control'] == 4, result['specialist'] == 4)
    return result


class SAMClient:
    """Persistent, isolated SAM3 process; only own RGB and masks cross pipes."""

    def __init__(self, model_dir, device, python=None):
        self.scratch = tempfile.TemporaryDirectory(prefix='eurs-s79-sam-', dir='/tmp')
        self.child = None
        self.closed = False
        scratch = Path(self.scratch.name)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                   HF_HOME=str(scratch / 'hf'), XDG_CACHE_HOME=str(scratch / 'cache'),
                   TORCH_HOME=str(scratch / 'torch'), TMPDIR=str(scratch),
                   TRITON_CACHE_DIR=str(scratch / 'triton'),
                   TORCHINDUCTOR_CACHE_DIR=str(scratch / 'inductor'),
                   TORCH_EXTENSIONS_DIR=str(scratch / 'extensions'))
        executable = python or os.environ.get('EURS_SAM_PYTHON') or sys.executable
        command = [str(executable), '-u', '-B', str(Path(__file__).with_name('sam_worker.py')),
                   '--model-dir', str(Path(model_dir).resolve()), '--device', str(device)]
        try:
            self.child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=None, cwd=str(scratch), env=env)
            frame_write(self.child.stdin, {'command': 'initialize'})
            self.ready, data = frame_read(self.child.stdout, timeout=300)
            if self.ready.get('status') != 'READY' or data:
                raise RuntimeError(f'SAM3 initialization failed: {self.ready}')
        except BaseException:
            self.close()
            raise

    def predict(self, rgb):
        assert not self.closed and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
        h, w = rgb.shape[:2]
        assert 0 < h <= 1024 and 0 < w <= 1024
        content = rgb.tobytes()
        digest = hashlib.sha256(content).hexdigest()
        frame_write(self.child.stdin, dict(command='predict', height=h, width=w,
                                           rgb_sha256=digest), content)
        header, data = frame_read(self.child.stdout, timeout=300)
        if header.get('status') != 'PASS':
            raise RuntimeError(f'SAM3 prediction failed: {header}')
        assert header['rgb_sha256'] == digest and (header['height'], header['width']) == (h, w)
        assert len(data) == (2*h*w + 7) // 8
        support, boxes = np.unpackbits(np.frombuffer(data, np.uint8), count=2*h*w).reshape(2, h, w).astype(bool)
        return support, boxes

    def close(self):
        if self.closed:
            return
        self.closed = True
        child = self.child
        try:
            if child is not None:
                if child.poll() is None:
                    try:
                        frame_write(child.stdin, {'command': 'quit'})
                    except (BrokenPipeError, OSError):
                        pass
                if child.stdin is not None:
                    child.stdin.close()
                try:
                    child.wait(timeout=30)
                finally:
                    if child.stdout is not None:
                        child.stdout.close()
        finally:
            self.scratch.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Segmenter:
    """Own-image S79 masks; trained models are loaded only on construction."""

    def __init__(self, model_dir, device='cuda', sam_python=None):
        self.model_dir = Path(model_dir).resolve()
        self.device = torch.device(device)
        self.sam_python = sam_python
        self.sam = None
        self.closed = False
        # SAM3 changes these flags only inside its separate process.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.field = R15Field(self.model_dir)
        self.models = [load_component(name, root=self.model_dir, device=self.device)
                       for name in ['dinov3_positive', 'dinov3_background']]
        # Submission19433 changes ONLY this original EURS classifier. All
        # frozen encoder, LoRA, DPT and native heads remain identical to19386.
        classifier = torch.load(self.model_dir / 'weights/positive_classifier.pt',
                                map_location='cpu', weights_only=True)
        assert set(classifier) == {'weight', 'bias'}
        assert classifier['weight'].shape == (5, 256, 1, 1)
        assert classifier['bias'].shape == (5,)
        self.models[0].eurs.load_state_dict(classifier, strict=True)
        self.animals = load_component('mask2former_animals', root=self.model_dir, device=self.device)
        self.background = load_component('mask2former_background', root=self.model_dir, device=self.device)

    @torch.inference_mode()
    def __call__(self, path):
        if self.closed:
            raise RuntimeError('Segmenter is closed')
        sample = decode(path)
        rgb = self.field.predict(sample['values'], sample['stem'])
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None]
        first = dino_probabilities(self.models[0], x, self.device)
        second = dino_probabilities(self.models[1], x, self.device)
        raw = torch.from_numpy(sample['rgb']).permute(2, 0, 1)[None]
        second = (second + dino_probabilities(self.models[1], raw, self.device)) * .5
        first = (first * .5 + second * .5)[0].cpu().numpy()
        assert np.isfinite(first).all() and np.allclose(first.sum(0), 1, atol=1e-5)
        with _autocast(self.device):
            output = self.animals(pixel_values=normalize(x, self.device), output_auxiliary_logits=False)
        animals = m2f_scores(output, rgb.shape[:2])[0].cpu().numpy()
        del output
        with _autocast(self.device):
            output = self.background(pixel_values=normalize(x, self.device), output_auxiliary_logits=False)
        background = m2f_scores(output, rgb.shape[:2])[0].cpu().numpy()
        del output
        if sample['nativezero']:
            support = np.zeros(rgb.shape[:2], bool)
            boxes = support.copy()
        else:
            if self.sam is None:
                self.sam = SAMClient(self.model_dir, self.device, self.sam_python)
            support, boxes = self.sam.predict(rgb)
        mask = compose(first, animals, background, support, boxes, sample['nativezero'])['specialist']
        assert mask.dtype == np.uint8 and mask.shape == sample['rgb'].shape[:2] and mask.max() < 5
        return mask

    def close(self):
        if not self.closed:
            self.closed = True
            if self.sam is not None:
                self.sam.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

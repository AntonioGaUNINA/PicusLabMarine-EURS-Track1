"""Isolated local SAM3 image/text worker; no server registry or file proposals.

Run with the separately pinned SAM3 interpreter when primary Torch dependencies
do not support SAM3. Complete CUDA parity and cold memory have not been tested.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import struct
import sys
import time
import traceback


MAX_FRAME_BYTES = 4 * 1024 * 1024


def _exact(stream, count, timeout=None):
    deadline = None if timeout is None else time.monotonic() + timeout
    chunks = []
    remaining = count
    # os.read avoids BufferedReader read-ahead defeating readiness checks.
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError):
        descriptor = None
    while remaining:
        if deadline is not None and descriptor is not None:
            with selectors.DefaultSelector() as selector:
                selector.register(descriptor, selectors.EVENT_READ)
                if not selector.select(max(0, deadline - time.monotonic())):
                    raise TimeoutError('Timed out waiting for SAM3 worker')
        content = os.read(descriptor, remaining) if descriptor is not None else stream.read(remaining)
        if not content:
            raise EOFError('SAM3 pipe closed before complete frame')
        chunks.append(content)
        remaining -= len(content)
    return b''.join(chunks)


def frame_write(stream, header, data=b''):
    raw = json.dumps(dict(header, payload_bytes=len(data)), separators=(',', ':')).encode()
    if len(raw) > MAX_FRAME_BYTES or len(data) > MAX_FRAME_BYTES:
        raise ValueError('SAM3 frame exceeds bounded own-image size')
    stream.write(struct.pack('!I', len(raw)))
    stream.write(raw)
    stream.write(data)
    stream.flush()


def frame_read(stream, timeout=None):
    size = struct.unpack('!I', _exact(stream, 4, timeout))[0]
    if not 0 < size <= MAX_FRAME_BYTES:
        raise ValueError('Invalid SAM3 header size')
    header = json.loads(_exact(stream, size, timeout))
    payload_bytes = header['payload_bytes']
    if not isinstance(payload_bytes, int) or not 0 <= payload_bytes <= MAX_FRAME_BYTES:
        raise ValueError('Invalid SAM3 payload size')
    return header, _exact(stream, payload_bytes, timeout)


def support_and_boxes(processor, rgb):
    """Exact composition_runtime.sam_server mask/box union in prompt order."""
    import numpy as np
    from PIL import Image
    h, w = rgb.shape[:2]
    support = np.zeros((h, w), bool)
    boxes = np.zeros_like(support)
    state = processor.set_image(Image.fromarray(rgb))
    for prompt in ['starfish', 'seahorse', 'sea turtle']:
        processor.reset_all_prompts(state)
        result = processor.set_text_prompt(prompt=prompt, state=state)
        masks = result['masks'][:, 0].cpu().numpy()
        bounds = result['boxes'].float().cpu().tolist()
        assert masks.shape[1:] == (h, w)
        if len(masks):
            support |= masks.any(0)
        for x0, y0, x1, y1 in bounds:
            left, top = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
            right, bottom = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
            if right > left and bottom > top:
                boxes[top:bottom, left:right] = True
    return support, boxes


def run_worker(model_dir, device):
    output = sys.stdout.buffer
    sys.stdout = sys.stderr  # Reserve stdout exclusively for framed binary IO.
    initialize, payload = frame_read(sys.stdin.buffer)
    if initialize.get('command') != 'initialize' or payload:
        raise ValueError('Expected SAM3 initialization frame')
    import numpy as np
    import torch
    # Executed directly by a separate interpreter: import this local module only.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from components import load_component
    torch.set_num_threads(8)
    torch.set_num_interop_threads(2)
    torch.manual_seed(2027)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.device(device).type != 'cuda':
        raise RuntimeError('The preserved SAM3 builder requires CUDA; CPU model parity is unverified')
    # Importing the original builder enables TF32 on Ampere, matching S79.
    model = load_component('sam3', root=Path(model_dir), device=device)
    from sam3.model.sam3_image_processor import Sam3Processor
    if not torch.backends.cuda.matmul.allow_tf32 or not torch.backends.cudnn.allow_tf32:
        raise RuntimeError('SAM3 historical Ampere TF32 policy was not enabled by its builder')
    processor = Sam3Processor(model, device=device, resolution=1008, confidence_threshold=.5)
    frame_write(output, dict(status='READY', torch=torch.__version__, input='own RGB bytes only',
                             observed_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                             observed_cudnn_tf32=torch.backends.cudnn.allow_tf32))
    while True:
        header, data = frame_read(sys.stdin.buffer)
        if header.get('command') == 'quit':
            break
        h, w = header['height'], header['width']
        assert header['command'] == 'predict' and 0 < h <= 1024 and 0 < w <= 1024
        assert len(data) == h*w*3 and hashlib.sha256(data).hexdigest() == header['rgb_sha256']
        rgb = np.frombuffer(data, np.uint8).reshape(h, w, 3).copy()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            support, boxes = support_and_boxes(processor, rgb)
        payload = np.packbits(np.stack([support, boxes]).reshape(-1)).tobytes()
        frame_write(output, dict(status='PASS', height=h, width=w,
                                 rgb_sha256=header['rgb_sha256'],
                                 peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                                 peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30), payload)
    frame_write(output, {'status': 'closed'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    try:
        run_worker(args.model_dir, args.device)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        # A structured failure is useful when construction fails in another env.
        try:
            frame_write(sys.__stdout__.buffer, dict(status='ERROR', error=type(error).__name__,
                                                    message=str(error)))
        except (BrokenPipeError, OSError):
            pass
        raise SystemExit(1)


if __name__ == '__main__':
    main()

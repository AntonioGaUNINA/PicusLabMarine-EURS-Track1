"""Exact own-image inference reproduction for EURS Track1 submission19433."""
import os
from pathlib import Path
import random
import sys
import tempfile
from unittest.mock import patch

sys.dont_write_bytecode = True


def main(model_dir, input_dir, output_dir, device):
    """Read own degraded PNGs and regenerate the submitted JPEG99/PNG pairs."""
    model_dir, input_dir, output_dir = map(Path, (model_dir, input_dir, output_dir))
    images = sorted(input_dir.rglob('*.png'))
    if not images:
        raise ValueError('No degraded PNG images found')
    stems = [p.stem for p in images]
    if len(stems) != len(set(stems)):
        raise ValueError('Input filename stems must be unique')
    for path in images:
        _, a, b = path.stem.rsplit('_', 2)
        if not (1 <= int(a) <= 10 and int(b) in range(5, 101, 5)):
            raise ValueError('Expected numeric water type and level suffix')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError('Use a fresh output directory to preserve prior results')
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ['restored', 'seg_masks']:
        (output_dir / directory).mkdir()

    with tempfile.TemporaryDirectory(prefix='eurs19433-') as scratch_name:
        scratch = Path(scratch_name)
        for key, directory in [('HF_HOME', 'hf'), ('TORCH_HOME', 'torch'),
                ('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                ('TORCH_EXTENSIONS_DIR', 'extensions'), ('XDG_CACHE_HOME', 'xdg')]:
            os.environ[key] = str(scratch / directory)
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        import numpy as np
        from PIL import Image
        import torch
        from src import official_resize_for_submission as official_resize
        from src.restoration import Restoration
        from src.segmentation import Segmenter

        random.seed(2027)
        np.random.seed(2027)
        torch.manual_seed(2027)
        if torch.device(device).type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is required for the submitted inference path')
            torch.cuda.manual_seed_all(2027)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Match the submitted inference runtime even if an outer checker set
        # deterministic=True before calling this entry point.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        torch.set_num_threads(8)
        restoration = Restoration(model_dir, device=device)
        with Segmenter(model_dir, device=device) as segmentation:
            for image_path in images:
                rgb = restoration(image_path)
                mask = segmentation(image_path)
                if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                    raise ValueError('Restoration must return uint8 HWC RGB')
                if mask.dtype != np.uint8 or mask.shape != rgb.shape[:2] or mask.max() > 4:
                    raise ValueError('Segmentation must return own-image uint8 class IDs0–4')

                # One image at a time bounds scratch usage and preserves the
                # official helper as the final spatial operation, even at size1024.
                with tempfile.TemporaryDirectory(prefix='image-', dir=scratch) as one_name:
                    one = Path(one_name)
                    for directory in ['degraded', 'rgb', 'mask']:
                        (one / directory).mkdir()
                    stem = image_path.stem
                    (one / 'degraded' / (stem + '.png')).symlink_to(image_path.resolve())
                    Image.fromarray(rgb).save(one / 'rgb' / (stem + '.png'), compress_level=1)
                    Image.fromarray(mask).save(one / 'mask' / (stem + '.png'), compress_level=1)
                    argv = ['resize_for_submission.py', '--degraded-dir', str(one / 'degraded'),
                        '--restored-dir', str(one / 'rgb'), '--masks-dir', str(one / 'mask'),
                        '--output-dir', str(one / 'resized')]
                    with patch.object(sys, 'argv', argv):
                        official_resize.main()
                    with Image.open(one / 'resized/restored' / (stem + '.png')) as photo:
                        photo.save(output_dir / 'restored' / (stem + '.jpg'), format='JPEG',
                                   quality=99, subsampling=0, optimize=True)
                    (output_dir / 'seg_masks' / (stem + '.png')).write_bytes(
                        (one / 'resized/seg_masks' / (stem + '.png')).read_bytes())


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    main(args.model_dir, args.input_dir, args.output_dir, args.device)

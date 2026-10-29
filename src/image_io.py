"""Own-image decoding, retaining the historical fractional F-channel resize."""
from pathlib import Path

import numpy as np
from PIL import Image

from .official_resize_for_submission import scale_to_max_side


def decode(path):
    """Match infer_eurs.decode without its research/controller imports."""
    path = Path(path)
    assert path.suffix.lower() == '.png' and path.is_file()
    parts = path.stem.rsplit('_', 2)
    assert len(parts) == 3 and 1 <= int(parts[-2]) <= 10
    assert int(parts[-1]) in range(5, 101, 5)
    with Image.open(path) as image:
        assert image.mode == 'RGB', (path, image.mode)
        original_size = image.size
        size = scale_to_max_side(*image.size, 1024)
        black = all(lo == hi == 0 for lo, hi in image.getextrema())
        rgb = np.array(image.resize(size, Image.Resampling.BILINEAR), copy=True)
        native = np.array(image, copy=True)
    values = np.stack([
        np.asarray(Image.fromarray(native[:, :, c].astype(np.float32)).resize(
            size, Image.Resampling.BILINEAR)) for c in range(3)
    ], -1)
    assert values.shape == rgb.shape and values.dtype == np.float32
    assert np.isfinite(values).all()
    return dict(input=str(path.resolve()), stem=path.stem, scene=parts[0],
                score_size=list(size), native_size=list(original_size),
                nativezero=black, rgb=rgb, values=values)

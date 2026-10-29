#!/usr/bin/env python3
"""Resize restored images and segmentation masks to the exact resolution the
EURS evaluator scores at, before zipping a submission.

The evaluator does NOT resize submissions to match ground truth -- an output
whose pixel dimensions don't exactly match what it expects is rejected
outright, not silently corrected. The expected size for each scene is
computed from that scene's downloaded DEGRADED image dimensions (which match
the ground truth's native resolution), using the exact same formula the
evaluator itself uses (see eurs_evaluator.py's _scale_to_max_side): scale the
longest side down to MAX_SIDE, round both dimensions with Python's round(),
never upscale.

Usage:
    python3 resize_for_submission.py \\
        --degraded-dir /path/to/downloaded/degraded \\
        --restored-dir /path/to/your/raw/restored/outputs \\
        --masks-dir /path/to/your/raw/seg_mask/outputs \\
        --output-dir ./submission \\
        --zip-output ./submission.zip

This produces <output-dir>/restored/ and <output-dir>/seg_masks/ at exactly
the resolution the evaluator expects. Pass --zip-output to also package them
into a single .zip (with restored/ and seg_masks/ at the archive root, no
extra nesting) ready to upload as-is -- otherwise zip those two folders
yourself the same way before submitting.

Run this as the last step of your pipeline, always -- even if you believe
your model already outputs at the right resolution. It is a no-op (a
straight copy) for any image that's already the correct size.
"""
import argparse
import os
import os.path as osp
import sys
import zipfile

from PIL import Image

MAX_SIDE = 1024  # must match eurs_evaluator.py's RANKED_EVAL_MAX_SIDE


def scale_to_max_side(w, h, max_side):
    """Identical to eurs_evaluator.py's _scale_to_max_side -- do not change
    this independently of that function, the two must always agree."""
    scale = max_side / max(w, h)
    if scale >= 1:
        return w, h
    return max(1, round(w * scale)), max(1, round(h * scale))


def find_input(directory, stem):
    if not osp.isdir(directory):
        return None
    for fname in os.listdir(directory):
        if osp.splitext(fname)[0] == stem:
            return osp.join(directory, fname)
    return None


def zip_submission(output_dir, zip_path):
    """Zip <output_dir>/restored and <output_dir>/seg_masks at the archive
    root, matching the format the evaluator expects."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for folder in ('restored', 'seg_masks'):
            folder_path = osp.join(output_dir, folder)
            for fname in sorted(os.listdir(folder_path)):
                zf.write(osp.join(folder_path, fname), osp.join(folder, fname))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--degraded-dir', required=True,
                         help="Folder of the original downloaded degraded images "
                              "(used only to read each scene's native resolution).")
    parser.add_argument('--restored-dir', required=True,
                         help='Folder of your restored-image outputs, at any resolution.')
    parser.add_argument('--masks-dir', required=True,
                         help='Folder of your segmentation mask outputs, at any resolution.')
    parser.add_argument('--output-dir', required=True,
                         help='Where to write the correctly-sized restored/ and seg_masks/ folders.')
    parser.add_argument('--zip-output', default=None,
                         help='If set, also package the resized output into a single .zip '
                              'at this path, ready to upload as-is.')
    args = parser.parse_args()

    out_restored = osp.join(args.output_dir, 'restored')
    out_masks = osp.join(args.output_dir, 'seg_masks')
    os.makedirs(out_restored, exist_ok=True)
    os.makedirs(out_masks, exist_ok=True)

    degraded_files = sorted(os.listdir(args.degraded_dir))
    if not degraded_files:
        sys.exit(f"No files found in --degraded-dir {args.degraded_dir}")

    missing_restored, missing_masks, done = [], [], 0
    for fname in degraded_files:
        stem = osp.splitext(fname)[0]
        with Image.open(osp.join(args.degraded_dir, fname)) as deg_img:
            target_w, target_h = scale_to_max_side(deg_img.width, deg_img.height, MAX_SIDE)

        restored_src = find_input(args.restored_dir, stem)
        if restored_src is None:
            missing_restored.append(stem)
        else:
            with Image.open(restored_src) as img:
                img = img.convert('RGB')
                if img.size != (target_w, target_h):
                    img = img.resize((target_w, target_h), Image.BILINEAR)
                img.save(osp.join(out_restored, f'{stem}.png'))

        mask_src = find_input(args.masks_dir, stem)
        if mask_src is None:
            missing_masks.append(stem)
        else:
            with Image.open(mask_src) as img:
                if img.size != (target_w, target_h):
                    img = img.resize((target_w, target_h), Image.NEAREST)
                img.save(osp.join(out_masks, f'{stem}.png'))

        if restored_src is not None and mask_src is not None:
            done += 1

    print(f"Resized {done}/{len(degraded_files)} entries into {args.output_dir}")
    if missing_restored:
        print(f"WARNING: {len(missing_restored)} restored output(s) not found, e.g.: {missing_restored[:5]}")
    if missing_masks:
        print(f"WARNING: {len(missing_masks)} mask output(s) not found, e.g.: {missing_masks[:5]}")
    if missing_restored or missing_masks:
        print("Your submission will be rejected as incomplete until every degraded image has both outputs.")
        sys.exit(1)

    if args.zip_output:
        zip_submission(args.output_dir, args.zip_output)
        print(f"Wrote {args.zip_output} -- ready to upload.")


if __name__ == '__main__':
    main()

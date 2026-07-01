import base64
import io

import numpy as np
from PIL import Image


def np_to_base64(img_array):
    if len(img_array.shape) == 2:
        img = Image.fromarray(img_array.astype(np.uint8), mode='L')
    else:
        img = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def split_and_filter_patches(image_array, patch_size=256, dark_threshold=8.0, min_variance=2.0, max_preview_patches=20):
    if image_array is None:
        return {
            "patch_size": patch_size,
            "total_patches": 0,
            "kept_patches": 0,
            "removed_patches": 0,
            "patch_summaries": [],
            "preview_patches": []
        }

    if len(image_array.shape) == 3:
        gray = np.mean(image_array, axis=-1).astype(np.float32)
    else:
        gray = image_array.astype(np.float32)

    if gray.size == 0:
        return {
            "patch_size": patch_size,
            "total_patches": 0,
            "kept_patches": 0,
            "removed_patches": 0,
            "patch_summaries": [],
            "preview_patches": []
        }

    target_h = int(np.ceil(gray.shape[0] / patch_size) * patch_size)
    target_w = int(np.ceil(gray.shape[1] / patch_size) * patch_size)
    pad_h = target_h - gray.shape[0]
    pad_w = target_w - gray.shape[1]
    padded = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

    patch_summaries = []
    preview_patches = []
    total_patches = 0
    removed_patches = 0

    for y in range(0, padded.shape[0], patch_size):
        for x in range(0, padded.shape[1], patch_size):
            patch = padded[y:y + patch_size, x:x + patch_size]
            total_patches += 1

            patch_mean = float(np.mean(patch))
            patch_std = float(np.std(patch))
            nonzero_ratio = float(np.count_nonzero(patch > 0) / patch.size)
            is_dark = patch_mean <= dark_threshold or (patch_std <= min_variance and nonzero_ratio < 0.05)

            if is_dark:
                removed_patches += 1
                continue

            summary = {
                "x": int(x),
                "y": int(y),
                "width": int(patch.shape[1]),
                "height": int(patch.shape[0]),
                "mean": round(patch_mean, 4),
                "std": round(patch_std, 4),
                "nonzero_ratio": round(nonzero_ratio, 4),
            }
            patch_summaries.append(summary)
            if len(preview_patches) < max_preview_patches:
                preview_patches.append({
                    **summary,
                    "image": np_to_base64(patch.astype(np.uint8))
                })

    return {
        "patch_size": patch_size,
        "total_patches": total_patches,
        "kept_patches": len(patch_summaries),
        "removed_patches": removed_patches,
        "patch_summaries": patch_summaries,
        "preview_patches": preview_patches,
    }

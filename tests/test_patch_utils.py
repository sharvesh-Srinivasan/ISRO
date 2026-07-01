import numpy as np

from patch_utils import split_and_filter_patches


def test_split_and_filter_patches_removes_dark_tiles():
    image = np.zeros((120, 120), dtype=np.uint8)
    image[0:60, 0:60] = 255

    result = split_and_filter_patches(image, patch_size=60, dark_threshold=10.0, min_variance=1.0)

    assert result["total_patches"] == 4
    assert result["kept_patches"] == 1
    assert result["removed_patches"] == 3
    assert result["patch_summaries"][0]["x"] == 0
    assert result["patch_summaries"][0]["y"] == 0

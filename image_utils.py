import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform


def apply_mask_to_image(
    image_path: str, mask_weight: np.ndarray, scale: int = 24, text: str = None, smooth: bool = True
):

    image = Image.open(image_path)
    mask_height, mask_width = mask_weight.shape
    image = image.resize([mask_height * scale, mask_width * scale], Image.LANCZOS)
    plt.imshow(image)

    if smooth:
        mask_weight = skimage.transform.pyramid_expand(mask_weight, upscale=scale, sigma=8)
    else:
        mask_weight = skimage.transform.resize(mask_weight, [mask_height * scale, mask_width * scale])

    plt.imshow(mask_weight, alpha=0.8)

    plt.set_cmap(cm.Greys_r)
    plt.axis("off")

    if text:
        plt.text(0, 1, text, color="black", backgroundcolor="white", fontsize=12)

"""Classical (model-free) image features built on Pillow + NumPy."""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from ..registry import feature


def _arr(img): return np.asarray(img.convert("RGB"), dtype=np.float32)
def _img(a): return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


@feature("enhance", "Image Enhancement", "Auto contrast, color and sharpness.")
def enhance(img, strength=1.0):
    img = ImageOps.autocontrast(img.convert("RGB"), cutoff=1)
    img = ImageEnhance.Color(img).enhance(1 + 0.2 * strength)
    return ImageEnhance.Sharpness(img).enhance(1 + 0.5 * strength)


@feature("upscale", "Super-resolution upscaling", "Lanczos upscale with unsharp masking.")
def upscale(img, scale=2):
    big = img.convert("RGB").resize((img.width * scale, img.height * scale), Image.LANCZOS)
    return big.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))


@feature("remove_bg", "Image Remove BG", "Removes a background similar to the border color.")
def remove_bg(img, tolerance=40.0):
    a = _arr(img)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(a - bg, axis=2)
    alpha = np.clip((dist - tolerance) / max(tolerance, 1) * 255, 0, 255).astype(np.uint8)
    out = img.convert("RGBA")
    out.putalpha(Image.fromarray(alpha).filter(ImageFilter.GaussianBlur(1)))
    return out


@feature("whiten", "Whitening & Shaping", "Brightens skin-toned regions (skin whitening only).")
def whiten(img, amount=0.25):
    a = _arr(img)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    skin = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & (abs(r - g) > 15)
    mask = Image.fromarray((skin * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(4))
    m = (np.asarray(mask, dtype=np.float32) / 255)[..., None]
    lifted = a + (255 - a) * amount
    return _img(a * (1 - m) + lifted * m)


@feature("remove_watermark", "Watermark removal", "Fills a user-supplied mask by diffusion inpainting.")
def remove_watermark(img, mask, iterations=200):
    a = _arr(img)
    m = np.asarray(mask.convert("L").resize(img.size)) > 127
    out = a.copy()
    out[m] = a[~m].mean(axis=0) if (~m).any() else 0
    for _ in range(iterations):
        p = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        avg = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4
        out[m] = avg[m]
    return _img(out)


@feature("colorize", "Colorize Photos", "Maps luminance onto a warm/cool palette (non-learned).")
def colorize(img, shadow=(30, 40, 80), highlight=(255, 225, 190)):
    g = np.asarray(ImageOps.grayscale(img), dtype=np.float32)[..., None] / 255
    lo, hi = np.array(shadow, np.float32), np.array(highlight, np.float32)
    return _img(lo + (hi - lo) * g)


@feature("old_photo_repair", "Old Photo Repair", "Median despeckle, contrast restore, mild sharpen.")
def old_photo_repair(img):
    img = img.convert("RGB").filter(ImageFilter.MedianFilter(3))
    img = ImageOps.autocontrast(img, cutoff=2)
    return img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=60))


@feature("change_sky", "Change sky", "Replaces the bright blue-ish top region with a chosen sky image/color.")
def change_sky(img, sky):
    a = _arr(img)
    h = a.shape[0]
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    cand = (b > r + 10) & (b > 120)
    cand[int(h * 0.6):] = False
    mask = Image.fromarray((cand * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3))
    m = (np.asarray(mask, np.float32) / 255)[..., None]
    if isinstance(sky, tuple):
        s = np.full_like(a, 0) + np.array(sky, np.float32)
    else:
        s = _arr(sky.resize(img.size))
    return _img(a * (1 - m) + s * m)

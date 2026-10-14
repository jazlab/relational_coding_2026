r"""**The luminance contrast of each object identity against the task's background.**

The task shows three object identities, ``'a'``, ``'b'`` and ``'c'``. They differ in colour
and therefore in how bright they are against the noisy grey background, and
:class:`models_lib.WeightedAverageModel` asks whether a neuron's response to a pair is
weighted by that difference. This module computes the number that model needs.

What is measured
----------------
Per the specification: **the average luminance of the object — including the white circle it
is rendered on — against the average luminance of the background.**

An object on screen is a **white disc** of radius ``PREY_SCALE`` with the identity's image
composited on top, and the background is a noisy grey field. So

    L_obj(id) = mean luminance over the disc of {white circle + object image}
    L_bg      = mean luminance of the background field
    contrast  = (L_obj - L_bg) / (L_obj + L_bg)          (Michelson)

Why this is a reimplementation, and how faithful it is
------------------------------------------------------
``../../data_exploration/task_visualization/renderer.py`` is the authority on how a frame looks, but it
imports ``moog``, which is not installed here. Rather than approximate, the two steps that
matter are reproduced **exactly** from that file, and the constants are read from it:

* **The background** (``Renderer.__init__``): load ``background_noise.npy``, take
  ``clip(3n - 1, 0, 1)``, resize 2x with bilinear interpolation, rescale so the mean is
  ``BACKGROUND_VALUE``, clip, quantise to 8 bits. Grey, so luminance is the pixel value.

* **The object** (``Renderer._render_prey_patch``): the identity's JPEG resized to
  ``floor((sqrt(2) + 0.37) * PREY_SCALE * image_size)`` pixels square and flipped top to
  bottom, then pasted over the white disc **through a mask that makes every pixel with all
  three channels above 220 transparent**. That mask is the subtle part and it is not
  cosmetic: the JPEGs have a white surround, and the square is large enough that its corners
  fall outside the disc — the mask is what keeps them from being drawn, and what lets the
  white circle show through the surround.

The disc itself is drawn at full opacity during the stimulus phase
(``task_state.get_task_state`` gives the prey sprite ``c0=0, c1=0, c2=1`` — white — and
``trial_structure.ipynb`` sets ``opacity = 255``), so nothing of the background shows through
it and ``L_obj`` does not depend on where the object happens to sit.

Luminance
---------
Rec. 709 luma, ``0.2126 R + 0.7152 G + 0.0722 B``, on the 8-bit values scaled to [0, 1] — the
standard weighting for perceived brightness. :func:`object_contrast` also reports the flat
channel mean, which is the same story to two decimal places.
"""

import functools
from pathlib import Path

import numpy as np
from PIL import Image

#: Copied from ``data_exploration/task_visualization/renderer.py`` — the values the task was run with.
PREY_SCALE = 0.083  #: object radius, in units of the screen's [0, 1] extent
BACKGROUND_VALUE = 0.3  #: the background's mean luminance, by construction
PATCH_FACTOR = np.sqrt(2) + 0.37  #: the object image is this many radii across
WHITE_CUTOFF = (
    220  #: a patch pixel above this in *all* channels is transparent
)
IMAGE_SIZE = 512  #: the renderer's default frame size, in pixels

#: Rec. 709 luma weights.
LUMA = np.array([0.2126, 0.7152, 0.0722])

OBJECT_IDS = ("a", "b", "c")

_TASK_VIS = (
    Path(__file__).resolve().parents[2]
    / "data_exploration"
    / "task_visualization"
)
_RESOURCES = _TASK_VIS / "resources"


def luminance(rgb):
    """Rec. 709 luma of an RGB array scaled to [0, 1]."""
    return np.asarray(rgb, float) @ LUMA


@functools.lru_cache(maxsize=1)
def background_image(image_size=IMAGE_SIZE):
    """The task's background field as a float array in [0, 1] — ``Renderer.__init__``'s
    construction, reproduced step for step."""
    noise = np.load(_RESOURCES / "background_noise.npy")
    height, width = noise.shape
    noise = np.clip(3.0 * noise - 1.0, 0.0, 1.0)
    noise = np.array(
        Image.fromarray(noise).resize(
            (2 * height, 2 * width), resample=Image.BILINEAR
        )
    )
    noise = (BACKGROUND_VALUE / np.mean(noise)) * noise
    noise = np.clip(noise, 0.0, 1.0)
    return (255 * noise).astype(np.uint8) / 255.0


@functools.lru_cache(maxsize=8)
def object_patch(object_id, image_size=IMAGE_SIZE):
    """``(rgb, opaque)`` for one identity's image: the resized, flipped JPEG in [0, 1], and
    the boolean mask of the pixels the renderer actually draws."""
    size = int(np.floor(PATCH_FACTOR * PREY_SCALE * image_size))
    im = Image.open(_RESOURCES / "objects" / f"{object_id}.jpg")
    im = im.resize((size, size)).transpose(Image.FLIP_TOP_BOTTOM)
    rgb8 = np.asarray(im)
    opaque = ~np.all(rgb8 > WHITE_CUTOFF, axis=2)
    return rgb8 / 255.0, opaque


@functools.lru_cache(maxsize=8)
def object_composite(object_id, image_size=IMAGE_SIZE):
    """``(rgb, disc)`` — the object as it appears on screen, and the disc it occupies.

    The canvas is the white circle; the identity's image is composited onto it through the
    renderer's mask. Returned at patch resolution with the disc marked, which is all the
    contrast needs.
    """
    rgb, opaque = object_patch(object_id, image_size)
    size = rgb.shape[0]
    canvas = np.ones_like(rgb)  # the white circle
    canvas[opaque] = rgb[opaque]

    # the disc, in the patch's own pixel grid
    radius_px = PREY_SCALE * image_size
    centre = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    disc = ((xx - centre) ** 2 + (yy - centre) ** 2) <= radius_px**2
    return canvas, disc


def object_contrast(object_id, image_size=IMAGE_SIZE):
    """Every quantity behind one identity's contrast, as a dict, so the working can be
    inspected rather than the number taken on trust."""
    canvas, disc = object_composite(object_id, image_size)
    _, opaque = object_patch(object_id, image_size)
    lum = luminance(canvas)
    bg = background_image(image_size)

    l_obj = float(lum[disc].mean())
    l_bg = float(bg.mean())
    return dict(
        object_id=object_id,
        l_obj=l_obj,
        l_bg=l_bg,
        michelson=(l_obj - l_bg) / (l_obj + l_bg),
        weber=(l_obj - l_bg) / l_bg,
        l_obj_flat=float(
            canvas[disc].mean()
        ),  # unweighted channel mean, for checking
        frac_disc_covered=float(
            opaque[disc].mean()
        ),  # how much of the disc is the object
        n_disc_px=int(disc.sum()),
    )


@functools.lru_cache(maxsize=1)
def contrasts(image_size=IMAGE_SIZE):
    """``{object_id: Michelson contrast}`` for the three identities."""
    return {k: object_contrast(k, image_size)["michelson"] for k in OBJECT_IDS}


def relative_contrast(id_a, id_b, image_size=IMAGE_SIZE):
    r"""The **signed relative contrast** of object A against object B,

        d = (C_A - C_B) / (C_A + C_B),

    which is what :class:`models_lib.WeightedAverageModel` weights each object's drive by.
    It is antisymmetric — ``d(B, A) = -d(A, B)`` — and scale-free, so it does not matter
    whether contrast is measured in Michelson units or any other positive multiple of them.
    Zero when the two objects are equally visible, which the task never shows: the two
    objects on a trial always have different identities.
    """
    c = contrasts(image_size)
    ca = np.asarray([c[i] for i in np.atleast_1d(id_a)], float)
    cb = np.asarray([c[i] for i in np.atleast_1d(id_b)], float)
    tot = ca + cb
    out = np.where(
        np.abs(tot) > 1e-12,
        (ca - cb) / np.where(np.abs(tot) > 1e-12, tot, 1.0),
        0.0,
    )
    return out if np.ndim(id_a) else float(out[0])

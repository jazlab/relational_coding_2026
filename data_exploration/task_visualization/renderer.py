"""Python Image Library (PIL/Pillow) renderer for MOOG game states."""

from pathlib import Path

import numpy as np
from moog.observers import color_maps
from PIL import Image, ImageDraw

# Resolve paths to resources
_RESOURCES_PATH = Path(__file__).parent / "resources"
_FOVEAL_WINDOW_PATH = _RESOURCES_PATH / "foveal_window.npy"
_NOISE_PATH = _RESOURCES_PATH / "background_noise.npy"
_OBJECTS_PATH = _RESOURCES_PATH / "objects"

# Constants for the task. Do not change these. These values were used in the
# original task design.
_CUE_SCALE = 0.16
_PREY_SCALE = 0.083
_BACKGROUND_VALUE = 0.3


class Renderer:
    """Render using Python Image Library (PIL/Pillow).

    This renders an environment state as an image.
    """

    def __init__(self, image_size=(512, 512)):
        """Construct renderer.

        Args:
            image_size: Size of the rendered image in pixels.
        """
        self._image_size = image_size

        # Create prey patches
        object_ids = [
            "strawberry",
            "apple",
            "banana",
            "a",
            "b",
            "c",
            "kiwi",
            "plum",
            "tomato",
        ]
        image_patches = {
            k: Image.open(_OBJECTS_PATH / f"{k}.jpg") for k in object_ids
        }
        prey_size_pixels = int(
            np.floor((np.sqrt(2) + 0.37) * _PREY_SCALE * image_size[0])
        )
        prey_patches = {
            k: v.resize((prey_size_pixels, prey_size_pixels)).transpose(
                Image.FLIP_TOP_BOTTOM
            )
            for k, v in image_patches.items()
        }
        self._prey_size_pixels = prey_size_pixels
        self._prey_patches = prey_patches

        # Create cue patches
        rescale = 0.5 * _CUE_SCALE / _PREY_SCALE
        cue_size_pixels = int(rescale * list(prey_patches.values())[0].size[0])
        self._cue_patches = {
            k: v.resize((cue_size_pixels, cue_size_pixels))
            for k, v in prey_patches.items()
        }

        # Initialize canvas and drawing context
        self._canvas = Image.new("RGB", self._image_size)
        self._draw = ImageDraw.Draw(self._canvas, "RGBA")
        self._canvas_prey = Image.new("RGBA", self._image_size)
        self._draw_prey = ImageDraw.Draw(self._canvas_prey, "RGBA")
        self._prey_bg = Image.new("RGB", self._image_size, (0, 0, 0))
        self._fw_raw = Image.fromarray(np.load(_FOVEAL_WINDOW_PATH))

        # Load background noise
        noise = np.load(_NOISE_PATH)
        height, width = noise.shape
        noise_clipped = np.clip(3 * noise - 1.0, 0.0, 1.0)
        noise = Image.fromarray(noise_clipped)
        noise = noise.resize((2 * height, 2 * width), resample=Image.BILINEAR)
        noise = np.array(noise)
        noise = (_BACKGROUND_VALUE / np.mean(noise)) * noise
        noise = np.clip(noise, 0.0, 1.0)
        noise = (255 * noise).astype(np.uint8)
        self._bg = np.stack([noise, noise, noise], axis=2)

    def _get_fw_mask(self, state):
        self._fw = Image.new("L", self._image_size, 0)
        fw = state["fw"][0]
        fw_radius = fw.metadata["radius"]
        patch_size = int(2.0 * fw_radius * self._image_size[0])
        patch = self._fw_raw.resize((patch_size, patch_size))
        patch_pos = fw.position * self._image_size[0]
        patch_pos = patch_pos - patch.size[0] / 2
        self._fw.paste(patch, tuple(patch_pos.astype(int)))

    def _render_bg(self, background_indices):
        ih, iw = background_indices
        bg = self._bg[
            ih : ih + self._image_size[0], iw : iw + self._image_size[1]
        ]
        canvas_bg = Image.fromarray(bg)
        self._canvas.paste(canvas_bg)

    def _draw_polygon(self, draw, sprite, opacity=None):
        vertices = self._image_size * sprite.vertices
        color = color_maps.hsv_to_rgb(sprite.color)
        if opacity is None:
            opacity = sprite.opacity
        color = tuple(list(color) + [opacity])
        draw.polygon([tuple(v) for v in vertices], fill=color)

    def _render_prey_patch(self, sprite, canvas, opacity=None):
        prey_id = sprite.metadata["id"]
        if prey_id is None:
            return
        patch = self._prey_patches[sprite.metadata["id"]]

        patch_white = np.prod(np.array(patch) > 220, axis=2)
        if opacity is None:
            opacity = sprite.opacity
        mask = opacity * (1.0 - patch_white.astype(int))
        mask = Image.fromarray(mask.astype(np.uint8), mode="L")
        p_half_width = 0.5 * np.array(patch.size)
        patch_pos = np.round(self._image_size * sprite.position - p_half_width)
        canvas.paste(patch, tuple(patch_pos.astype(int)), mask)

    def _render_cue_patch(self, sprite, state, canvas, opacity=None):
        if sprite.metadata["target_prey_ind"] is None:
            return

        prey_sprite = state["prey"][sprite.metadata["target_prey_ind"]]
        patch_id = prey_sprite.metadata["id"]
        if patch_id is None:
            return
        patch = self._cue_patches[patch_id]

        patch_white = np.prod(np.array(patch) > 220, axis=2)
        if opacity is None:
            opacity = sprite.opacity
        mask = opacity * (1.0 - patch_white.astype(int))
        mask = Image.fromarray(mask.astype(np.uint8), mode="L")
        p_half_width = 0.5 * np.array(patch.size)
        patch_pos = np.round(self._image_size * sprite.position - p_half_width)
        canvas.paste(patch, tuple(patch_pos.astype(int)), mask)

    def _render_prey(self, state):
        self._canvas_prey.paste(self._prey_bg)

        for sprite in state["prey"]:
            self._draw_polygon(self._draw_prey, sprite, opacity=255)
            self._render_prey_patch(sprite, self._canvas_prey, opacity=255)

        for sprite in state["cue"]:
            self._draw_polygon(self._draw_prey, sprite)
            self._render_cue_patch(sprite, state, self._canvas_prey)

    def __call__(self, state, background_indices, blank=False):
        """Render sprites.

        The order of layers in the state is background to foreground, and the
        order of sprites within layers is also background to foreground.

        Args:
            state: OrderedDict of iterables of sprites.

        Returns:
            Numpy uint8 RGB array of size self._image_size + (3,).
        """
        self._render_bg(background_indices)
        self._get_fw_mask(state)

        for layer in state:
            if layer == "fw":
                continue
            for sprite in state[layer]:
                self._draw_polygon(self._draw, sprite)
                if layer == "prey":
                    self._render_prey_patch(sprite, self._canvas)
                if layer == "cue":
                    self._render_cue_patch(sprite, state, self._canvas)

        if state["fw"][0].opacity == 255:
            self._render_prey(state)
            self._canvas.paste(self._canvas_prey, (0, 0), self._fw)

        if blank:
            for sprite in state["prey"]:
                self._draw_polygon(self._draw, sprite, opacity=255)

        if state["eye"][0].opacity == 255:
            self._draw_polygon(self._draw, state["eye"][0])

        if state["broke_fixation_screen"][0].opacity == 255:
            self._draw_polygon(self._draw, state["broke_fixation_screen"][0])

        image = self._canvas.resize(self._image_size, resample=Image.LANCZOS)

        # PIL uses a coordinate system with the origin (0, 0) at the upper-left,
        # but our environment uses an origin at the bottom-left (i.e.
        # mathematical convention). Hence we need to flip the render vertically
        # to correct for that.
        image = np.flipud(np.array(image))

        return image

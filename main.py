#!/usr/bin/env python3

from math import pi
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from dataclasses import dataclass
from typing import List

type Vector2 = np.ndarray
type Image = np.ndarray
type MatrixNx = np.ndarray
type Matrixx = np.ndarray

PATCH_SIZE = 10
PATCH_RANGE = lambda: range(-PATCH_SIZE // 2, PATCH_SIZE // 2 + 1)
PATCH: MatrixNx = np.array([[x, y] for x in PATCH_RANGE() for y in PATCH_RANGE()])
ZOOM_PAD = 32
ROT_CIRCLE = np.array([0, PATCH_SIZE])  # Debug circle to show the rotation

IMG1 = "img1.png"
IMG2 = "img2.png"
C1 = np.array([222.919, 137.575])
C2 = np.array([513.16, 168.145])

img1 = plt.imread(IMG1)
img2 = plt.imread(IMG2)


@dataclass
class DrawingState:
    fig: plt.Figure
    ax: plt.Axes
    img: Image
    circles: List[plt.Circle]
    rot_circle: plt.Circle
    center: Vector2
    angle: float

    def redraw_angle(self, angle):
        self.angle = angle
        tpatch = PATCH @ R(self.angle).T + self.center
        for circle, center in zip(self.circles, tpatch):
            circle.center = center
        self.rot_circle.center = R(self.angle) @ [0, PATCH_SIZE] + self.center
        print(E(C1, C2, img1, img2, angle))


def R(angle: float) -> Matrixx:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def E(c1: Vector2, c2: Vector2, img1: Image, img2: Image, angle: float) -> float:
    tpatch1 = PATCH + c1
    tpatch2 = PATCH @ R(angle).T + c2

    tpatch1 = np.round(tpatch1).astype(int)
    tpatch2 = np.round(tpatch2).astype(int)

    # Notice the switch in x and y
    intensities1 = img1[tpatch1[:, 1], tpatch1[:, 0]]
    intensities2 = img2[tpatch2[:, 1], tpatch2[:, 0]]
    return np.sum((intensities1 - intensities2) ** 2)


def get_match_rotation(c1: Vector2, c2: Vector2) -> float:
    return pi / 6


def make_drawing(img_file: str, c: Vector2, angle: float) -> DrawingState:
    img = plt.imread(img_file)
    fig, ax = plt.subplots()
    ax.set_xlim(c[0] - ZOOM_PAD, c[0] + ZOOM_PAD)
    ax.set_ylim(c[1] + ZOOM_PAD, c[1] - ZOOM_PAD)
    ax.imshow(img, cmap="gray")
    ax.plot(c[0], c[1], "r+")
    tpatch = PATCH @ R(angle).T + c
    circle_kwargs = {"facecolor": "none", "edgecolor": "r"}
    circles = [plt.Circle(p, 0.5, **circle_kwargs) for p in tpatch]
    for circle in circles:
        ax.add_patch(circle)
    rot_circle = plt.Circle(R(angle) @ ROT_CIRCLE + c, 0.5, **circle_kwargs)
    ax.add_patch(rot_circle)
    state = DrawingState(fig, ax, img, circles, rot_circle, c, angle)
    return state


def main():
    make_drawing(IMG1, C1, 0)

    # Reference patch
    angle = get_match_rotation(C1, C2)
    drawing = make_drawing(IMG2, C2, angle)

    # Optimized patch
    axangle = drawing.fig.add_axes([0.25, 0.025, 0.55, 0.03])
    freq_slider = Slider(ax=axangle, label="Angle θ", valmin=-pi, valmax=pi, valinit=0)
    freq_slider.on_changed(drawing.redraw_angle)

    plt.show()


if __name__ == "__main__":
    main()

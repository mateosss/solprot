#!/usr/bin/env python3

from math import pi, sin, cos
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from dataclasses import dataclass
from typing import List
from bilinterp import interp, interp_grad, intensity
import PIL.Image

type Vector2 = np.ndarray
type VectorN = np.ndarray
type Image = np.ndarray
type MatrixNx2 = np.ndarray
type Matrix2x2 = np.ndarray

PATCH_SIZE = 10
PATCH_RANGE = lambda: range(-PATCH_SIZE // 2, PATCH_SIZE // 2 + 1)
PATCH: MatrixNx2 = np.array([[x, y] for x in PATCH_RANGE() for y in PATCH_RANGE()])
ZOOM_PAD = 32
ROT_CIRCLE = np.array([0, PATCH_SIZE])  # Debug circle to show the rotation

IMG1 = "img1.png"
IMG2 = "img2.png"
# C1 = np.array([222.919, 137.575])
# C2 = np.array([513.16, 168.145])
C1 = np.array([321.298, 209.150])
C2 = np.array([598.566, 110.105])
img1 = plt.imread(IMG1)
img2 = plt.imread(IMG2)
img1_raw = np.array(PIL.Image.open(IMG1))
img2_raw = np.array(PIL.Image.open(IMG2))


@dataclass
class DrawingState:
    fig: plt.Figure
    ax: plt.Axes
    img: Image
    circles: List[plt.Circle]
    rot_circle: plt.Circle
    center: Vector2
    angle: float
    scale: float

    def redraw(self, angle, scale):
        tpatch = scale * PATCH @ R(angle).T + self.center
        for circle, center in zip(self.circles, tpatch):
            circle.center = center
        self.rot_circle.center = R(angle) @ [0, PATCH_SIZE] + self.center
        self.fig.canvas.draw_idle()
        print(
            f"Redraw: [OLD] E(θ={self.angle:.2f}, s={self.scale:.2f}) = {E(self.angle, self.scale):.2f}"
        )
        print(f"        [NEW] E(θ={angle:.2f}, s={scale}) = {E(angle, scale):.2f}")
        self.angle = angle
        self.scale = scale

    def iterate_angle(self, event) -> float:
        # tpatch1 = np.round(PATCH + C1).astype(int)
        # tpatch2 = np.round(PATCH @ R(self.angle).T + C2).astype(int)
        tpatch1 = PATCH + C1
        tpatch2 = self.scale * PATCH @ R(self.angle).T + C2
        i1 = np.array([interp(img1_raw, p[0], p[1]) for p in tpatch1])
        i2 = np.array([interp(img2_raw, p[0], p[1]) for p in tpatch2])
        Jr = J_r(self.angle, self.scale)
        Jr_pinv = np.linalg.inv(Jr.T @ Jr) @ Jr.T
        new_angle, new_scale = Jr_pinv @ (i2 - i1)
        self.redraw(new_angle, new_scale)


def R(angle: float) -> Matrix2x2:
    return np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])


def J_R(angle: float) -> Matrix2x2:
    return np.array([[-sin(angle), -cos(angle)], [cos(angle), -sin(angle)]])


def J_r(angle: float, scale: float) -> VectorN:
    N = PATCH.shape[0]
    rows = []
    for i in range(N):
        point: Vector2 = PATCH[i]
        R_deriv: Matrix2x2 = J_R(angle)
        I_deriv: Vector2 = np.array([0, 0], dtype=np.float32)
        tpoint = scale * R(angle) @ point
        interp_grad(img2_raw, tpoint[0], tpoint[1], I_deriv)
        J_r_angle = scale * (I_deriv @ R_deriv @ point)
        J_r_scale = scale * I_deriv @ R(angle) @ point
        rows.append([J_r_angle, J_r_scale])
    rows = np.array(rows)
    return rows


# def r_linearized(angle: float) -> VectorN:
#     tpatch1 = np.round(PATCH + C1).astype(int)
#     tpatch2 = np.round(PATCH @ R(angle).T + C2).astype(int)
#     i1 = np.array([interp(img1_raw, p[0], p[1]) for p in tpatch1])
#     i2 = np.array([interp(img2_raw, p[0], p[1]) for p in tpatch2])
#     return i1 - i2 + J_r(angle) * angle


def r(angle: float, scale: float) -> VectorN:
    tpatch1 = PATCH + C1
    tpatch2 = scale * PATCH @ R(angle).T + C2
    # tpatch1 = np.round(tpatch1).astype(int)
    # tpatch2 = np.round(tpatch2).astype(int)

    # Column swap because images have shape HxW, but tpatch addresses x,y
    # intensities1 = img1[tpatch1[:, 1], tpatch1[:, 0]]
    # intensities2 = img2[tpatch2[:, 1], tpatch2[:, 0]]
    intensities1 = np.array([interp(img1_raw, p[0], p[1]) for p in tpatch1])
    intensities2 = np.array([interp(img2_raw, p[0], p[1]) for p in tpatch2])
    return intensities1 - intensities2


def E(angle: float, scale: float) -> float:
    return np.sum(r(angle, scale) ** 2)


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
    state = DrawingState(fig, ax, img, circles, rot_circle, c, angle, 1)
    return state


def main():
    make_drawing(IMG1, C1, 0)

    # Reference patch
    angle = pi / 6
    drawing = make_drawing(IMG2, C2, angle)

    # Optimized patch
    angle_ax = drawing.fig.add_axes([0.25, 0.025, 0.55, 0.03])
    angle_slider = Slider(ax=angle_ax, label="Angle θ", valmin=-pi, valmax=pi, valinit=0)
    angle_slider.on_changed(lambda angle: drawing.redraw(angle, drawing.scale))
    scale_ax = drawing.fig.add_axes([0.25, 0.05, 0.55, 0.03])
    scale_slider = Slider(ax=scale_ax, label="Scale s", valmin=0.1, valmax=2.5, valinit=1)
    scale_slider.on_changed(lambda scale: drawing.redraw(drawing.angle, scale))
    axbtn = drawing.fig.add_axes([0.025, 0.025, 0.1, 0.05])
    opt_btn = Button(axbtn, "Step")
    opt_btn.on_clicked(drawing.iterate_angle)

    plt.show()


if __name__ == "__main__":
    main()

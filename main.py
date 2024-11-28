#!/usr/bin/env python3

# https://answers.opencv.org/question/232700/calculate-orb-descriptors-for-arbitrary-image-points/

from math import pi, sin, cos, e
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.optimize import brute, differential_evolution
import PIL.Image

from bilinterp import interp, interp_grad, intensity

type Vector2 = np.ndarray
type VectorN = np.ndarray
type Image = np.ndarray
type MatrixNx2 = np.ndarray
type Matrix2x2 = np.ndarray


def get_square_patch(patch_size=10):
    "Square patch"
    prange = lambda: range(-patch_size // 2, patch_size // 2 + 1)
    patch: MatrixNx2 = np.array([[x, y] for x in prange() for y in prange()])
    rot_circle = np.array([0, patch_size])
    return patch, rot_circle


def get_squared_patch_cubically_condensed(patch_size_p=10, patch_radius_px=10):
    "Square patch condensed towards the center with a cubic function"
    f = lambda x: x**3
    prange = lambda: f(np.linspace(-1, 1, patch_size_p)) * patch_radius_px
    patch: MatrixNx2 = np.array([[x, y] for x in prange() for y in prange()])
    rot_circle = np.array([0, patch_radius_px * 2.5])
    return patch, rot_circle


def get_squared_patch_log_condensed(PATCH_RADIUS_P=3, patch_radius_px=10):
    "Square patch condensed towards the center with a inverse sigmoid function"
    patch_size_p = 2 * PATCH_RADIUS_P + 1
    f = lambda x: np.log(1 / (x / 2 + 0.5) - 1) / -5  # Attract towards 0
    ONE = (1 / (e**-5 + 1) - 0.5) * 2  # f(ONE) == 1
    prange = lambda: f(np.linspace(-ONE, ONE, patch_size_p)) * patch_radius_px
    patch: MatrixNx2 = np.array([[x, y] for x in prange() for y in prange()])
    rot_circle = np.array([0, patch_radius_px * 2.5])
    return patch, rot_circle


def get_circle_patch(RADIUS=5, ANGLES=6, SCALE=2):
    "Circle patch"
    patch = np.array(
        [
            [cos(a) * s * SCALE, sin(a) * s * SCALE]
            for s in np.linspace(1, RADIUS, RADIUS)
            for a in np.linspace(0, 2 * pi, int(ANGLES * s), endpoint=False)
        ]
    )
    rot_circle = np.array([0, RADIUS * 2.5])
    return patch, rot_circle


# fmt: off
EXAMPLES = [
    ("images/img1.png", "images/img2.png", np.array([321.298, 209.150]), np.array([598.566, 110.105])),
    ("images/img1.png", "images/img2.png", np.array([222.919, 137.575]), np.array([513.16, 168.145])),
    ("images/aprilgrid.png", "images/aprilgrid15.png", np.array([148.775, 137.947]), np.array([182.983, 114.219])),
    ("images/aprilgrid.png", "images/aprilgrid60.png", np.array([364.5, 67.5]), np.array([470.68, 254.27])),
]
# fmt: on

ZOOM_PAD = 32
IMG1, IMG2, C1, C2 = EXAMPLES[3]
PATCH, ROT_CIRCLE = get_square_patch()

img1 = plt.imread(IMG1)
img2 = plt.imread(IMG2)
img1_raw = np.array(PIL.Image.open(IMG1))
img2_raw = np.array(PIL.Image.open(IMG2))


@dataclass
class DrawingState:
    fig: plt.Figure
    ax: plt.Axes
    img: Image  # float32 0-1
    img_raw: Image  # uint8 0-255
    circles: List[plt.Circle]
    rot_circle: plt.Circle
    center: Vector2
    angle: float
    fill: str = "None"

    def set_fill(self, fill: str):
        if fill == "None":
            for circle in self.circles:
                circle.set_facecolor("none")
                circle.set_edgecolor("r")
        elif fill in ["Sample", "Grad", "Reference", "Residual"]:
            for circle in self.circles:
                circle.set_edgecolor("none")
        else:
            raise ValueError(f"Unknown fill '{self.fill}'")
        self.fill = fill
        self.update_fill()

    def update_fill(self):
        if self.fill == "Sample":
            for circle in self.circles:
                color = interp(self.img_raw, circle.center[0], circle.center[1])
                circle.set_facecolor(f"{color}")
        if self.fill == "Grad":
            for circle in self.circles:
                grad: Vector2 = np.array([0, 0], dtype=np.float32)
                color = interp_grad(
                    self.img_raw, circle.center[0], circle.center[1], grad
                )
                grad = grad / 2 + 0.5  # Normalize from [-1, 1] to [0, 1]
                # circle.set_facecolor((grad[0], grad[1], color))
                circle.set_facecolor((grad[0], grad[1], 0.0))
        if self.fill == "Reference":
            for circle in self.circles:
                center2 = circle.center
                center1 = R(self.angle).T @ (center2 - C2) + C1
                # color2 = interp(img2_raw, center2[0], center2[1])
                color1 = interp(img1_raw, center1[0], center1[1])
                circle.set_facecolor(f"{color1}")
        elif self.fill == "Residual":
            min_diff = -0.1
            max_diff = 0.1
            diffs = [0] * len(self.circles)
            for i, circle in enumerate(self.circles):
                center2 = circle.center
                center1 = R(self.angle).T @ (center2 - C2) + C1
                color2 = interp(img2_raw, center2[0], center2[1])
                color1 = interp(img1_raw, center1[0], center1[1])
                diff = color1 - color2
                min_diff = min(min_diff, diff)
                max_diff = max(max_diff, diff)
                diffs[i] = diff
            for diff, circle in zip(diffs, self.circles):
                v = abs(diff / max_diff) if diff >= 0 else -abs(diff / min_diff)
                v = v / 2 + 0.5
                circle.set_facecolor(colormaps["PiYG"](v))

    def redraw_angle(self, angle):
        tpatch = PATCH @ R(angle).T + self.center
        for circle, center in zip(self.circles, tpatch):
            circle.center = center
        self.rot_circle.center = R(angle) @ ROT_CIRCLE + self.center

        er0, er = E(self.angle), E(angle)
        l0, l = E_lin(self.angle, self.angle), E_lin(self.angle, angle)
        a0, a = self.angle, angle

        Jr = J_r(angle)
        Jr_pinv = Jr / (Jr @ Jr)  # moore-penrose pseudoinverse
        delta = -Jr_pinv @ r(angle)
        delta_gd = J_E(self.angle)

        delta_numeric = (E(self.angle + pi/180) - E(self.angle - pi/180)) / (2 * pi/180)
        self.text.set_text(
            f"E(θ={a:.2f})={er:.2f} <- NEW | LIN -> El(θ={a:.2f})={l:.2f} | next dθ[GN] = {delta:2f}\n"
            f"E(θ={a0:.2f})={er0:.2f} <- OLD | LIN -> El(θ={a0:.2f})={l0:.2f} | next dθ[GD] = {delta_gd:.2f} | next dθ[NUM] = {delta_numeric:.2f}\n"
        )

        self.angle = angle
        self.update_fill()
        self.fig.canvas.draw_idle()

    def iterate_angle_gn(self, _) -> float:  # Gauss newton
        Jr = J_r(self.angle)
        Jr_pinv = Jr / (Jr @ Jr)  # moore-penrose pseudoinverse
        delta = -Jr_pinv @ r(self.angle)
        new_angle = self.angle + delta
        self.redraw_angle(new_angle)

    def iterate_angle_scipy(self, _) -> float:
        # Brute global optimization (15ms for 100 iterations)
        res = brute(E, ((-pi, pi),), Ns=100, full_output=True, finish=None)
        new_angle = res[0]

        # Differential evolution global optimization (15ms)
        # res = differential_evolution(E, [(-pi, pi)], disp=True)
        # new_angle = res.x[0]

        self.redraw_angle(new_angle)

    lr = 0.1
    momentum = 0.5
    prev_delta = 0

    def iterate_angle_gd(self, _) -> float:
        delta = self.lr * J_E(self.angle) - self.momentum * self.prev_delta
        new_angle = self.angle - delta
        self.prev_delta = delta
        print(f"{J_E(self.angle)=:.5f}, {delta=:.5f}")
        self.redraw_angle(new_angle)


def R(angle: float) -> Matrix2x2:
    return np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])


def J_R(angle: float) -> Matrix2x2:
    return np.array([[-sin(angle), -cos(angle)], [cos(angle), -sin(angle)]])


def J_r(angle) -> VectorN:
    N = PATCH.shape[0]
    rows = []
    for i in range(N):
        point: Vector2 = PATCH[i]
        R_deriv: Matrix2x2 = J_R(angle)
        I_deriv: Vector2 = np.array([0, 0], dtype=np.float32)
        tpoint = R(angle) @ point
        interp_grad(img2_raw, tpoint[0], tpoint[1], I_deriv)
        row = I_deriv @ R_deriv @ point
        rows.append(row)
    rows = np.array(rows)
    return rows


def r_lin(angle_0: float, angle: float) -> VectorN:
    return r(angle_0) + J_r(angle_0) * (angle - angle_0)


def E_lin(angle_0: float, angle: float) -> float:
    return np.sum(r_lin(angle_0, angle) ** 2)


def r(angle: float) -> VectorN:
    tpatch1 = PATCH + C1
    tpatch2 = PATCH @ R(angle).T + C2
    i1 = np.array([interp(img1_raw, p[0], p[1]) for p in tpatch1])
    i2 = np.array([interp(img2_raw, p[0], p[1]) for p in tpatch2])
    return i1 - i2


def E(angle: float) -> float:
    return np.sum(r(angle) ** 2)


def J_E(angle: float) -> float:
    return 2 * (r(angle) @ J_r(angle))


def make_drawing(img_file: str, c: Vector2, angle: float) -> DrawingState:
    img = plt.imread(img_file)
    img_raw = np.array(PIL.Image.open(img_file))
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
    state = DrawingState(fig, ax, img, img_raw, circles, rot_circle, c, angle)
    return state


def main():
    make_drawing(IMG1, C1, 0)

    # Reference patch
    angle = pi / 6
    drawing = make_drawing(IMG2, C2, angle)

    # Optimized patch
    axangle = drawing.fig.add_axes([0.25, 0.025, 0.55, 0.03])
    freq_slider = Slider(ax=axangle, label="Angle θ", valmin=-pi, valmax=pi, valinit=0)
    freq_slider.on_changed(drawing.redraw_angle)

    axchecks = drawing.fig.add_axes([0.025, 0.25, 0.1, 0.2])
    check_labels = ["None", "Sample", "Grad", "Reference", "Residual"]
    actives = [True] + [False] * (len(check_labels) - 1)
    check = CheckButtons(ax=axchecks, labels=check_labels, actives=actives)

    def check_cb(label):
        print(f"Checked {label}")
        check.eventson = False
        check.clear()
        check.set_active(check_labels.index(label))
        drawing.set_fill(label)
        drawing.fig.canvas.draw_idle()
        check.eventson = True

    check.on_clicked(check_cb)

    axtext = drawing.fig.add_axes([0.2, 0.85, 0.2, 0.1])
    drawing.text = axtext.text(0, 0.5, "E(θ=___)=___", ha="left", va="bottom")
    axtext.axis("off")

    axscipy = drawing.fig.add_axes([0.025, 0.145, 0.1, 0.05])
    btnscipy = Button(axscipy, "Scipy")
    btnscipy.on_clicked(drawing.iterate_angle_scipy)

    axgd = drawing.fig.add_axes([0.025, 0.085, 0.1, 0.05])
    btngd = Button(axgd, "Step GD")
    btngd.on_clicked(drawing.iterate_angle_gd)

    axgn = drawing.fig.add_axes([0.025, 0.025, 0.1, 0.05])
    btngn = Button(axgn, "Step GN")
    btngn.on_clicked(drawing.iterate_angle_gn)

    plt.show()


if __name__ == "__main__":
    main()

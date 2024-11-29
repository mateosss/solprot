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

from bilinterp import batch_interp, batch_interp_grad

type Vector2 = np.ndarray
type VectorN = np.ndarray
type Image = np.ndarray
type MatrixNx2 = np.ndarray
type Matrix2x2 = np.ndarray

zeros_f32 = lambda n: np.zeros(n, dtype=np.float32)


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


def get_circle_patch(RADIUS=3, ANGLES=6, SCALE=1):
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
    ("images/MOO11A.png", "images/MOO11B.png", np.array([106.243, 331.75]), np.array([599.882, 313.076])),
    ("images/MOO11A.png", "images/MOO11B.png", np.array([18.012, 322.084]), np.array([487.170, 283.607])),
    ("images/MOO11A.png", "images/MOO11B.png", np.array([98.677, 266.376]), np.array([551.531, 234.406])),
]
# fmt: on

ZOOM_PAD = 32
IMG1, IMG2, C1, C2 = EXAMPLES[0]
PATCH, ROT_CIRCLE = get_circle_patch()
N = PATCH.shape[0]

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

    _slider: Slider = None
    _slider_ax: plt.Axes = None
    _checks: CheckButtons = None
    _checks_ax: plt.Axes = None
    _text: plt.Text = None
    _text_ax: plt.Axes = None
    _btn_global: Button = None
    _btn_global_ax: plt.Axes = None
    _btn_scipy: Button = None
    _btn_scipy_ax: plt.Axes = None
    _btn_gd: Button = None
    _btn_gd_ax: plt.Axes = None
    _btn_gn: Button = None
    _btn_gn_ax: plt.Axes = None

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
        centers = np.array([circle.center for circle in self.circles])
        valdxdys = zeros_f32((N, 3))
        batch_interp_grad(self.img_raw, centers[:, 0], centers[:, 1], out=valdxdys)
        colors, dxs, dys = valdxdys[:, 0], valdxdys[:, 1], valdxdys[:, 2]
        colors /= colors.mean()
        if self.fill == "Sample":
            a = colors.min()  # minimum mean-normalized color
            b = colors.max()  # maximum mean-normalized color
            for circle, color in zip(self.circles, colors):
                color = (color - a) / (b - a)  # minmax-normalized color
                circle.set_facecolor(f"{color}")
        if self.fill == "Grad":
            s = lambda x: (1 / (1 + e ** -(50 * (x - 0.5))))  # Push to edges
            f = lambda x: s(x / 2 + 0.5)  # Map [-1, 1] to [0, 1]
            for circle, dx, dy in zip(self.circles, dxs, dys):
                circle.set_facecolor((f(dx), f(dy), 0))
        if self.fill == "Reference":
            ref_centers = PATCH + C1
            colors1 = zeros_f32(N)
            batch_interp(img1_raw, ref_centers[:, 0], ref_centers[:, 1], out=colors1)
            colors1 /= colors1.mean()
            a = colors1.min()  # minimum mean-normalized color
            b = colors1.max()  # maximum mean-normalized color
            for circle, color1 in zip(self.circles, colors1):
                color1 = (color1 - a) / (b - a)  # minmax-normalized color
                circle.set_facecolor(f"{color1}")
        elif self.fill == "Residual":
            ref_centers = PATCH + C1
            colors1 = zeros_f32(N)
            batch_interp(img1_raw, ref_centers[:, 0], ref_centers[:, 1], out=colors1)
            colors1 /= colors1.mean()
            a1, b1 = colors1.min(), colors1.max()
            a2, b2 = colors.min(), colors.max()
            colors1 = (colors1 - a1) / (b1 - a1)
            colors = (colors - a2) / (b2 - a2)
            diffs = colors1 - colors
            assert diffs.max() < 1 and diffs.min() > -1, f"{diffs.max=}, {diffs.min=}"
            for diff, circle in zip(diffs, self.circles):
                circle.set_facecolor(colormaps["PiYG"](diff / 2 + 0.5))

    def redraw_angle(self, angle):
        tpatch = PATCH @ R(angle).T + self.center
        for circle, center in zip(self.circles, tpatch):
            circle.center = center
        self.rot_circle.center = R(angle) @ ROT_CIRCLE + self.center

        er0, er = E(self.angle), E(angle)
        l0, l = E_lin(self.angle, self.angle), E_lin(self.angle, angle)
        a0, a = self.angle, angle

        h = np.float64(0.001 * (2 * pi / 360))
        d_analytic = J_E(self.angle)
        d_numeric = (E(self.angle + h / 2) - E(self.angle - h / 2)) / h
        d_diff = np.linalg.norm(d_analytic - d_numeric)
        self._text.set_text(
            f"E(θ={a:.2f})={er:.2f} <- NEW | LIN -> El(θ={a:.2f})={l:.2f}\n"
            f"E(θ={a0:.2f})={er0:.2f} <- OLD | LIN -> El(θ={a0:.2f})={l0:.2f}\n"
            f"{d_analytic=:.4f} | {d_numeric=:.4f} | {d_diff=:.4f}"
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
        # Global optimization
        # new_angle = differential_evolution(E, [(-pi, pi)]).x[0] # ~5ms
        new_angle = brute(E, ((-pi, pi),), Ns=360, finish=None)  # ~5ms for Ns=100
        self.redraw_angle(new_angle)

    def iterate_global_opt(self, _) -> float:
        DIVS = 8
        STEPS = 5

        angles = np.linspace(0, 2 * pi, DIVS, endpoint=False)
        minerr_a = 0
        minerr = float("inf")
        for a in angles:
            for _ in range(STEPS):  # GN iteration
                Jr = J_r(a)
                Jr_pinv = Jr / (Jr @ Jr)
                delta = -Jr_pinv @ r(a)
                a += delta
            err = E(a)
            if err < minerr:
                minerr = err
                minerr_a = a
        self.redraw_angle(minerr_a)

    lr = 0.1
    momentum = 0.5
    prev_delta = 0

    def iterate_angle_gd(self, _) -> float:
        delta = self.lr * J_E(self.angle) - self.momentum * self.prev_delta
        new_angle = self.angle - delta
        self.prev_delta = delta
        self.redraw_angle(new_angle)

    def setup_ui(self):
        ax = self.fig.add_axes([0.25, 0.025, 0.55, 0.03])
        slider = Slider(ax=ax, label="Angle θ", valmin=-pi, valmax=pi, valinit=0)
        slider.on_changed(self.redraw_angle)
        self._slider = slider
        self._slider_ax = ax

        ax = self.fig.add_axes([0.025, 0.3, 0.1, 0.2])
        check_labels = ["None", "Sample", "Grad", "Reference", "Residual"]
        actives = [True] + [False] * (len(check_labels) - 1)
        checks = CheckButtons(ax=ax, labels=check_labels, actives=actives)

        def check_cb(label):
            print(f"Checked {label}")
            checks.eventson = False
            checks.clear()
            checks.set_active(check_labels.index(label))
            self.set_fill(label)
            self.fig.canvas.draw_idle()
            checks.eventson = True

        checks.on_clicked(check_cb)
        self._checks = checks
        self._checks_ax = ax

        ax = self.fig.add_axes([0.2, 0.85, 0.2, 0.1])
        text = ax.text(0, 0.5, "Awaiting for first E(θ)", ha="left", va="bottom")
        ax.axis("off")
        self._text = text
        self._text_ax = ax

        ax = self.fig.add_axes([0.025, 0.205, 0.1, 0.05])
        btn = Button(ax, "Global")
        btn.on_clicked(self.iterate_global_opt)
        self._btn_global = btn
        self._btn_global_axx = ax

        ax = self.fig.add_axes([0.025, 0.145, 0.1, 0.05])
        btn = Button(ax, "Scipy")
        btn.on_clicked(self.iterate_angle_scipy)
        self._btn_scipy = btn
        self._btn_scipy_ax = ax

        ax = self.fig.add_axes([0.025, 0.085, 0.1, 0.05])
        btn = Button(ax, "Step GD")
        btn.on_clicked(self.iterate_angle_gd)
        self._btn_gd = btn
        self._btn_gd_ax = ax

        ax = self.fig.add_axes([0.025, 0.025, 0.1, 0.05])
        btn = Button(ax, "Step GN")
        btn.on_clicked(self.iterate_angle_gn)
        self._btn_gn = btn
        self._btn_gn_ax = ax


def R(angle: float) -> Matrix2x2:
    return np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])


def J_R(angle: float) -> Matrix2x2:
    return np.array([[-sin(angle), -cos(angle)], [cos(angle), -sin(angle)]])


def J_r(angle: float) -> VectorN:
    R_deriv: Matrix2x2 = J_R(angle)
    tpoints = PATCH @ R(angle).T + C2
    valdxdys = zeros_f32((N, 3))
    batch_interp_grad(img2_raw, tpoints[:, 0], tpoints[:, 1], out=valdxdys)
    I_deriv = valdxdys[:, 1:3]
    derivs = I_deriv @ R_deriv
    res = np.einsum("ij,ij->i", derivs, PATCH)  # Perform dot product on each row
    constant = -1 / valdxdys[:, 0].mean()
    return constant * res


def r_lin(angle_0: float, angle: float) -> VectorN:
    return r(angle_0) + J_r(angle_0) * (angle - angle_0)


def E_lin(angle_0: float, angle: float) -> float:
    return np.sum(r_lin(angle_0, angle) ** 2)


def r(angle: float) -> VectorN:
    tpatch1 = PATCH + C1
    tpatch2 = PATCH @ R(angle).T + C2
    i1 = zeros_f32(N)
    i2 = zeros_f32(N)
    batch_interp(img1_raw, tpatch1[:, 0], tpatch1[:, 1], out=i1)
    batch_interp(img2_raw, tpatch2[:, 0], tpatch2[:, 1], out=i2)
    i1 /= i1.mean()
    i2 /= i2.mean()
    return i1 - i2


def E(angle: float) -> float:
    return np.sum(r(angle) ** 2)


def J_E(angle: float) -> float:
    return 2 * (r(angle) @ J_r(angle))


def make_drawing(
    img_file: str, c: Vector2, angle: float = 0, setup_ui: bool = False
) -> DrawingState:
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
    if setup_ui:
        state.setup_ui()
    return state


def main():
    np.set_printoptions(precision=4, suppress=True)

    make_drawing(IMG1, C1)

    # Unused variable to keep alive the UI
    drawing = make_drawing(IMG2, C2, setup_ui=True)  # pylint: disable=unused-variable

    plt.show()


if __name__ == "__main__":
    main()

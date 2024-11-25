#!/usr/bin/env python3

from math import pi
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

type Vector2 = np.ndarray
type Image = np.ndarray
type MatrixNx = np.ndarray
type Matrixx = np.ndarray

PATCH_SIZE = 10
PATCH_RANGE = lambda: range(-PATCH_SIZE // 2, PATCH_SIZE // 2 + 1)
PATCH: MatrixNx = np.array([[x, y] for x in PATCH_RANGE() for y in PATCH_RANGE()])


def R(angle: float) -> Matrixx:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def E(c1: Vector2, c2: Vector2, img1: Image, img2: Image, angle: float) -> float:
    tpatch1 = PATCH + c1
    tpatch2 = PATCH @ R(angle).T + c2

    tpatch1 = np.round(tpatch1).astype(int)
    tpatch2 = np.round(tpatch2).astype(int)

    intensities1 = img1[tpatch1[:, 0], tpatch1[:, 1]]
    intensities = img2[tpatch2[:, 0], tpatch2[:, 1]]
    return np.sum((intensities1 - intensities) ** 2)


def get_match_rotation(c1: Vector2, c2: Vector2, img1: Image, img: Image) -> float:
    return pi / 6


def draw(img: Image, c: Vector2, angle: float) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.plot(c[0], c[1], "r+")
    tpatch = PATCH @ R(angle).T + c
    for p in tpatch:
        circle = plt.Circle(p, 0.5, facecolor="none", edgecolor="r")
        ax.add_patch(circle)
    return fig, ax


def main():
    img1 = plt.imread("img1.png")
    c1 = np.array([222.919, 137.575])
    draw(img1, c1, 0)

    img2 = plt.imread("img2.png")
    c2 = np.array([513.16, 168.145])
    angle = get_match_rotation(c1, c2, img1, img1)
    draw(img2, c2, angle)

    plt.show()


if __name__ == "__main__":
    main()

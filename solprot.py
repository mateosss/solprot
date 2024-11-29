# Copyright 2024, Technical University of Munich
# SPDX-License-Identifier: BSD-3-Clause-Clear
# Author: Mateo de Mayo <mateo.demayo@tum.de>

from math import pi, cos, sin

import numpy as np
import PIL.Image
from scipy.optimize import brute

from bilinterp import batch_interp, batch_interp_grad


type Vector2 = np.ndarray  # float32, shape=(2,)
type Image = np.ndarray  # uint8, shape=(H, W)
type VectorN = np.ndarray
type Matrix2x2 = np.ndarray

# Use float32, float64 is overkill and slower
array = lambda x: np.array(x, dtype=np.float32)
zeros = lambda n: np.zeros(n, dtype=np.float32)


def get_circle_patch(RADIUS=3, ANGLES=6, SCALE=1):
    "Circle patch"
    patch = array(
        [
            [cos(a) * s * SCALE, sin(a) * s * SCALE]
            for s in np.linspace(1, RADIUS, RADIUS)
            for a in np.linspace(0, 2 * pi, int(ANGLES * s), endpoint=False)
        ]
    )
    return patch


PATCH = get_circle_patch()
N = PATCH.shape[0]

IMG1: Image = None
IMG2: Image = None
C1: Vector2 = None
C2: Vector2 = None


def R(angle: float) -> Matrix2x2:
    return array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])


def r(angle: float) -> VectorN:
    tpatch1 = PATCH + C1
    tpatch2 = PATCH @ R(angle).T + C2
    i1 = zeros(N)
    i2 = zeros(N)
    batch_interp(IMG1, tpatch1[:, 0], tpatch1[:, 1], out=i1)
    batch_interp(IMG2, tpatch2[:, 0], tpatch2[:, 1], out=i2)
    i1 /= i1.mean()
    i2 /= i2.mean()
    return i1 - i2


def E(angle: float) -> float:
    return np.sum(r(angle) ** 2)


def solve():
    return brute(E, ((-pi, pi),), Ns=360, finish=None)


def solve_patch_rotation(
    img0_fn: str, img1_fn: str, kp0: Vector2, kp1: Vector2
) -> float:
    """
    Return the rotation angle of a feature in IMG1 to align with img0
    :param img0_fn: path to reference image 0
    :param img1_fn: path to image 1 in which to estimate the feature rotation
    :param kp0: coordinates of the keypoint observation in img0
    :param kp1: coordinates of the keypoint observation in IMG1
    :return: rotation angle
    """

    # Using globals for speed: avoid self.- dereferencing
    global IMG1, IMG2, C1, C2  # pylint: disable=global-statement

    IMG1 = np.array(PIL.Image.open(img0_fn), dtype=np.uint8)
    IMG2 = np.array(PIL.Image.open(img1_fn), dtype=np.uint8)
    C1 = array(kp0)
    C2 = array(kp1)
    angle = solve()
    return angle

// Copyright 2024, Technical University of Munich
// SPDX-License-Identifier: BSD-3-Clause-Clear
// Author: Mateo de Mayo <mateo.demayo@tum.de>
// File adapted from https://gitlab.com/VladyslavUsenko/basalt-headers/-/blob/master/include/basalt/image/image.h
// Follow the link for the original license and copyright

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstdint>

#define V(x, y) image(y, x) / Scalar(255.0)

using arg = pybind11::arg;
namespace py = pybind11;

// TODO: Consider intensity is at pixel center and not left border as now
// TODO: Use Eigen for eigen-numpy pybind interoperability

using Scalar = float;

template <typename T>
using np_array = py::array_t<T>;

/// Image intensity value without bounds checking.
Scalar intensity(const np_array<uint8_t> image_arr, size_t x, size_t y) {
  py::buffer_info buf_info = image_arr.request();
  ssize_t w = buf_info.shape[1];
  uint8_t *ptr = static_cast<uint8_t *>(buf_info.ptr);
  return ptr[y * w + x] / Scalar(255.0);
}

/// Image intensity value from bilinear interpolation without bounds checking.
/// The interpolation accesses 4 pixels.
Scalar interp(const np_array<uint8_t> image_arr, Scalar x, Scalar y) {
  auto image = image_arr.unchecked<2>();

  size_t ix = x;
  size_t iy = y;

  Scalar dx = x - ix;
  Scalar dy = y - iy;

  Scalar ddx = Scalar(1.0) - dx;
  Scalar ddy = Scalar(1.0) - dy;

  Scalar px0y0 = V(ix, iy);
  Scalar px0y1 = V(ix, (iy + 1));
  Scalar px1y0 = V((ix + 1), iy);
  Scalar px1y1 = V((ix + 1), (iy + 1));

  Scalar p = ddx * ddy * px0y0 + ddx * dy * px0y1 + dx * ddy * px1y0 + dx * dy * px1y1;

  return p;
}

/// Image gradient from bilinear interpolation without bounds checking.
/// The gradient is computed using central differences (12 pixels are accessed).
Scalar interp_grad(const np_array<uint8_t> image_arr, Scalar x, Scalar y, np_array<float> out_dxdy_arr) {
  auto image = image_arr.unchecked<2>();
  auto out_dxdy = out_dxdy_arr.mutable_unchecked<1>();

  int ix = x;
  int iy = y;

  Scalar dx = x - ix;
  Scalar dy = y - iy;

  Scalar ddx = Scalar(1.0) - dx;
  Scalar ddy = Scalar(1.0) - dy;

  Scalar px0y0 = V(ix, iy);
  Scalar px0y1 = V(ix, iy + 1);
  Scalar px1y0 = V(ix + 1, iy);
  Scalar px1y1 = V(ix + 1, iy + 1);

  Scalar p = ddx * ddy * px0y0 + ddx * dy * px0y1 + dx * ddy * px1y0 + dx * dy * px1y1;

  Scalar pxm1y0 = V(ix - 1, iy);
  Scalar pxm1y1 = V(ix - 1, iy + 1);
  Scalar left = ddx * ddy * pxm1y0 + ddx * dy * pxm1y1 + dx * ddy * px0y0 + dx * dy * px0y1;

  Scalar px2y0 = V(ix + 2, iy);
  Scalar px2y1 = V(ix + 2, iy + 1);
  Scalar right = ddx * ddy * px1y0 + ddx * dy * px1y1 + dx * ddy * px2y0 + dx * dy * px2y1;

  out_dxdy(0) = Scalar(0.5) * (right - left);

  Scalar px0ym1 = V(ix, iy - 1);
  Scalar px1ym1 = V(ix + 1, iy - 1);
  Scalar top = ddx * ddy * px0ym1 + ddx * dy * px0y0 + dx * ddy * px1ym1 + dx * dy * px1y0;

  Scalar px0y2 = V(ix, iy + 2);
  Scalar px1y2 = V(ix + 1, iy + 2);
  Scalar bottom = ddx * ddy * px0y1 + ddx * dy * px0y2 + dx * ddy * px1y1 + dx * dy * px1y2;

  out_dxdy(1) = Scalar(0.5) * (bottom - top);

  return p;
}

/// Perform interp in batch
void batch_interp(const np_array<uint8_t> image_arr, const np_array<float> xs_arr, const np_array<float> ys_arr,
                  np_array<float> out_val_arr) {
  auto xs = xs_arr.unchecked<1>();
  auto ys = ys_arr.unchecked<1>();
  auto out_val = out_val_arr.mutable_unchecked<1>();

  for (py::ssize_t i = 0; i < xs.shape(0); i++) {
    out_val(i) = interp(image_arr, xs(i), ys(i));
  }
}

/// Perform interp_grad in batch
/// @todo Repeating same code as interp_grad, refactor
void batch_interp_grad(np_array<uint8_t> image_arr, const np_array<float> xs_arr, const np_array<float> ys_arr,
                       np_array<float> out_valdxdy_arr) {
  auto image = image_arr.unchecked<2>();
  auto xs = xs_arr.unchecked<1>();
  auto ys = ys_arr.unchecked<1>();
  auto out_valdxdy = out_valdxdy_arr.mutable_unchecked<2>();

  for (py::ssize_t i = 0; i < xs.shape(0); i++) {
    float x = xs(i);
    float y = ys(i);

    int ix = x;
    int iy = y;

    Scalar dx = x - ix;
    Scalar dy = y - iy;

    Scalar ddx = Scalar(1.0) - dx;
    Scalar ddy = Scalar(1.0) - dy;

    Scalar px0y0 = V(ix, iy);
    Scalar px0y1 = V(ix, iy + 1);
    Scalar px1y0 = V(ix + 1, iy);
    Scalar px1y1 = V(ix + 1, iy + 1);

    out_valdxdy(i, 0) = ddx * ddy * px0y0 + ddx * dy * px0y1 + dx * ddy * px1y0 + dx * dy * px1y1;

    Scalar pxm1y0 = V(ix - 1, iy);
    Scalar pxm1y1 = V(ix - 1, iy + 1);
    Scalar left = ddx * ddy * pxm1y0 + ddx * dy * pxm1y1 + dx * ddy * px0y0 + dx * dy * px0y1;

    Scalar px2y0 = V(ix + 2, iy);
    Scalar px2y1 = V(ix + 2, iy + 1);
    Scalar right = ddx * ddy * px1y0 + ddx * dy * px1y1 + dx * ddy * px2y0 + dx * dy * px2y1;

    out_valdxdy(i, 1) = Scalar(0.5) * (right - left);

    Scalar px0ym1 = V(ix, iy - 1);
    Scalar px1ym1 = V(ix + 1, iy - 1);
    Scalar top = ddx * ddy * px0ym1 + ddx * dy * px0y0 + dx * ddy * px1ym1 + dx * dy * px1y0;

    Scalar px0y2 = V(ix, iy + 2);
    Scalar px1y2 = V(ix + 1, iy + 2);
    Scalar bottom = ddx * ddy * px0y1 + ddx * dy * px0y2 + dx * ddy * px1y1 + dx * dy * px1y2;

    out_valdxdy(i, 2) = Scalar(0.5) * (bottom - top);
  }
}

PYBIND11_MODULE(bilinterp, m) {
  m.doc() = "Native image sampling with bilinear interpolation and gradients";
  m.def("intensity", &intensity, "Sample pixel at exact location", arg("img"), arg("x"), arg("y"));
  m.def("interp", &interp, "Sample pixel with bilinear interpolation", arg("img"), arg("x"), arg("y"));
  m.def("interp_grad", &interp_grad, "Same as interp, gradients are computed",  //
        arg("img"), arg("x"), arg("y"), arg("out"));
  m.def("batch_interp", &batch_interp, "Sample pixels with bilinear interpolation",  //
        arg("img"), arg("xs"), arg("ys"), arg("out"));
  m.def("batch_interp_grad", &batch_interp_grad, "Same as interp, gradients are computed",  //
        arg("img"), arg("xs"), arg("ys"), arg("out"));
}

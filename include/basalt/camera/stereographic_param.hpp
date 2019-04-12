/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <Eigen/Dense>

namespace basalt {

template <typename Scalar = double>
class StereographicParam {
 public:
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;

  static inline Vec2 project(const Vec4& p, Mat24* d_r_d_p = nullptr) {
    const Scalar sqrt = p.norm();
    const Scalar norm = p[2] + sqrt;
    const Scalar norm_inv = 1 / norm;

    const Vec2 res(p[0] * norm_inv, p[1] * norm_inv);

    if (d_r_d_p) {
      Scalar norm_inv2 = norm_inv * norm_inv;
      Scalar tmp = -norm_inv2 / sqrt;

      (*d_r_d_p).setZero();

      (*d_r_d_p)(0, 0) = norm_inv + p[0] * p[0] * tmp;
      (*d_r_d_p)(1, 0) = p[0] * p[1] * tmp;

      (*d_r_d_p)(1, 1) = norm_inv + p[1] * p[1] * tmp;
      (*d_r_d_p)(0, 1) = p[0] * p[1] * tmp;

      (*d_r_d_p)(0, 2) = p[0] * norm * tmp;
      (*d_r_d_p)(1, 2) = p[1] * norm * tmp;

      (*d_r_d_p)(0, 3) = 0;
      (*d_r_d_p)(1, 3) = 0;
    }

    return res;
  }

  static inline Vec4 unproject(const Vec2& p, Mat42* d_r_d_p = nullptr) {
    const Scalar x2 = p[0] * p[0];
    const Scalar y2 = p[1] * p[1];
    const Scalar r2 = x2 + y2;

    const Scalar norm_inv = Scalar(2) / (1 + r2);

    const Vec4 res(p[0] * norm_inv, p[1] * norm_inv, norm_inv - 1, 0);

    if (d_r_d_p) {
      const Scalar norm_inv2 = norm_inv * norm_inv;
      const Scalar xy = p[0] * p[1];

      (*d_r_d_p)(0, 0) = (norm_inv - x2 * norm_inv2);
      (*d_r_d_p)(0, 1) = -xy * norm_inv2;

      (*d_r_d_p)(1, 0) = -xy * norm_inv2;
      (*d_r_d_p)(1, 1) = (norm_inv - y2 * norm_inv2);

      (*d_r_d_p)(2, 0) = -p[0] * norm_inv2;
      (*d_r_d_p)(2, 1) = -p[1] * norm_inv2;

      (*d_r_d_p)(3, 0) = 0;
      (*d_r_d_p)(3, 1) = 0;
    }

    return res;
  }
};

}  // namespace basalt

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

#include <basalt/utils/sophus_utils.hpp>

namespace basalt {

template <typename Scalar = double>
class UnifiedCamera {
 public:
  static constexpr int N = 5;

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecN = Eigen::Matrix<Scalar, N, 1>;

  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;

  UnifiedCamera() { param.setZero(); }

  explicit UnifiedCamera(const VecN& p) { param = p; }

  template <class Scalar2>
  UnifiedCamera<Scalar2> cast() const {
    return UnifiedCamera<Scalar2>(param.template cast<Scalar2>());
  }

  static const std::string getName() { return "ucm"; }

  inline bool project(const Vec4& p3d, Vec2& proj,
                      Mat24* d_proj_d_p3d = nullptr,
                      Mat2N* d_proj_d_param = nullptr) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& alpha = param[4];

    const Scalar& x = p3d[0];
    const Scalar& y = p3d[1];
    const Scalar& z = p3d[2];

    const Scalar r2 = x * x + y * y;
    const Scalar rho2 = r2 + z * z;
    const Scalar rho = std::sqrt(rho2);

    // Check if valid
    const Scalar w = alpha > 0.5 ? (1 - alpha) / alpha : alpha / (1 - alpha);
    if (z <= -w * rho) return false;

    const Scalar norm = alpha * rho + (1 - alpha) * z;

    const Scalar mx = x / norm;
    const Scalar my = y / norm;

    proj = Vec2(fx * mx + cx, fy * my + cy);

    if (d_proj_d_p3d) {
      const Scalar denom = norm * norm * rho;
      const Scalar mid = -(alpha * x * y);
      const Scalar add = norm * rho;
      const Scalar addz = (alpha * z + (1 - alpha) * rho);

      (*d_proj_d_p3d)(0, 0) = fx * (add - x * x * alpha);
      (*d_proj_d_p3d)(1, 0) = fy * mid;
      (*d_proj_d_p3d)(0, 1) = fx * mid;
      (*d_proj_d_p3d)(1, 1) = fy * (add - y * y * alpha);
      (*d_proj_d_p3d)(0, 2) = -fx * x * addz;
      (*d_proj_d_p3d)(1, 2) = -fy * y * addz;
      (*d_proj_d_p3d)(0, 3) = 0;
      (*d_proj_d_p3d)(1, 3) = 0;

      (*d_proj_d_p3d) /= denom;
    }

    if (d_proj_d_param) {
      const Scalar norm2 = norm * norm;

      (*d_proj_d_param).setZero();
      (*d_proj_d_param)(0, 0) = mx;
      (*d_proj_d_param)(0, 2) = 1;
      (*d_proj_d_param)(1, 1) = my;
      (*d_proj_d_param)(1, 3) = 1;

      const Scalar tmp_x = -fx * x / norm2;
      const Scalar tmp_y = -fy * y / norm2;

      const Scalar tmp4 = (rho - z);

      (*d_proj_d_param)(0, 4) = tmp_x * tmp4;
      (*d_proj_d_param)(1, 4) = tmp_y * tmp4;
    }

    return true;
  }

  inline bool unproject(const Vec2& proj, Vec4& p3d,
                        Mat42* d_p3d_d_proj = nullptr,
                        Mat4N* d_p3d_d_param = nullptr) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& alpha = param[4];

    const Scalar& u = proj[0];
    const Scalar& v = proj[1];

    const Scalar xi = alpha / (1 - alpha);

    const Scalar mxx = (u - cx) / fx;
    const Scalar myy = (v - cy) / fy;

    const Scalar mx = (1 - alpha) * mxx;
    const Scalar my = (1 - alpha) * myy;

    const Scalar r2 = mx * mx + my * my;

    // Check if valid
    if (alpha > 0.5) {
      if (r2 >= Scalar(1.0) / ((2 * alpha - 1))) return false;
    }

    const Scalar xi2 = xi * xi;

    const Scalar n = std::sqrt(1 + (1 - xi2) * (r2));
    const Scalar m = (1 + r2);

    const Scalar k = (xi + n) / m;

    p3d[0] = k * mx;
    p3d[1] = k * my;
    p3d[2] = k - xi;
    p3d[3] = 0;

    if (d_p3d_d_proj || d_p3d_d_param) {
      const Scalar dk_dmx =
          -2 * mx * (n + xi) / (m * m) + mx * (1 - xi2) / (n * m);
      const Scalar dk_dmy =
          -2 * my * (n + xi) / (m * m) + my * (1 - xi2) / (n * m);

      Vec4 c0, c1;

      c0(0) = (dk_dmx * mx + k) / fx;
      c0(1) = dk_dmx * my / fx;
      c0(2) = dk_dmx / fx;
      c0(3) = 0;

      c1(0) = dk_dmy * mx / fy;
      c1(1) = (dk_dmy * my + k) / fy;
      c1(2) = dk_dmy / fy;
      c1(3) = 0;

      c0 *= (1 - alpha);
      c1 *= (1 - alpha);

      if (d_p3d_d_proj) {
        d_p3d_d_proj->col(0) = c0;
        d_p3d_d_proj->col(1) = c1;
      }

      if (d_p3d_d_param) {
        const Scalar d_xi_d_alpha = 1 / ((1 - alpha) * (1 - alpha));
        const Scalar d_m_d_alpha = -2 * (1 - alpha) * (mxx * mxx + myy * myy);

        const Scalar d_n_d_alpha = -(mxx * mxx + myy * myy) / n;

        const Scalar dk_d_alpha =
            ((d_xi_d_alpha + d_n_d_alpha) * m - d_m_d_alpha * (xi + n)) /
            (m * m);

        d_p3d_d_param->setZero();
        d_p3d_d_param->col(0) = -mxx * c0;
        d_p3d_d_param->col(1) = -myy * c1;
        d_p3d_d_param->col(2) = -c0;
        d_p3d_d_param->col(3) = -c1;

        (*d_p3d_d_param)(0, 4) = dk_d_alpha * mx - k * mxx;
        (*d_p3d_d_param)(1, 4) = dk_d_alpha * my - k * myy;
        (*d_p3d_d_param)(2, 4) = dk_d_alpha - d_xi_d_alpha;
      }
    }

    return true;
  }

  inline void setFromInit(const Vec4& init) {
    param[0] = init[0];
    param[1] = init[1];
    param[2] = init[2];
    param[3] = init[3];
    param[4] = 0.5;
  }

  void operator+=(const VecN& inc) { param += inc; }

  const VecN& getParam() const { return param; }

  static Eigen::vector<UnifiedCamera> getTestProjections() {
    Eigen::vector<UnifiedCamera> res;

    VecN vec1;

    // Euroc
    vec1 << 460.76484651566468, 459.4051018049483, 365.8937161309615,
        249.33499869752445, 0.5903365915227143;
    res.emplace_back(vec1);

    // TUM VI 512
    vec1 << 191.14799816648748, 191.13150946585135, 254.95857715233118,
        256.8815466235898, 0.6291060871161842;
    res.emplace_back(vec1);

    return res;
  }

  static Eigen::vector<Eigen::Vector2i> getTestResolutions() {
    Eigen::vector<Eigen::Vector2i> res;

    res.emplace_back(752, 480);
    res.emplace_back(512, 512);

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

}  // namespace basalt
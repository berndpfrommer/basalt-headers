/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2021, Collabora Ltd.
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

@file
@brief Implementation of pinhole camera model with radial-tangential distortion
@autor Mateo de Mayo <mateo.demayo@collabora.com>
*/

#pragma once

#include <basalt/camera/camera_static_assert.hpp>

#include <basalt/utils/sophus_utils.hpp>

namespace basalt {

using std::sqrt;

/// @brief Pinhole camera model with radial-tangential distortion
///
/// This model has N=12 parameters with \f$\mathbf{i} = \left[
/// f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6
/// \right]^T \f$. See \ref project and \ref unproject functions for more
/// details.
template <typename Scalar_ = double>
class PinholeRadtan8Camera {
 public:
  using Scalar = Scalar_;
  static constexpr int N = 12;  ///< Number of intrinsic parameters.

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecN = Eigen::Matrix<Scalar, N, 1>;

  using Mat22 = Eigen::Matrix<Scalar, 2, 2>;
  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;

  /// @brief Default constructor with zero intrinsics
  PinholeRadtan8Camera() { param_.setZero(); }

  /// @brief Construct camera model with given vector of intrinsics
  ///
  /// @param[in] p vector of intrinsic parameters [fx, fy, cx, cy, k1, k2, p1,
  /// p2, k3, k4, k5, k6]
  explicit PinholeRadtan8Camera(const VecN& p) { param_ = p; }

  /// @brief Cast to different scalar type
  template <class Scalar2>
  PinholeRadtan8Camera<Scalar2> cast() const {
    return PinholeRadtan8Camera<Scalar2>(param_.template cast<Scalar2>());
  }

  /// @brief Camera model name
  ///
  /// @return "pinhole-radtan8"
  static std::string getName() { return "pinhole-radtan8"; }

  /// @brief Project the point and optionally compute Jacobians
  ///
  /// Projection function is defined as follows:
  /// \f{align}{
  ///   \pi(\mathbf{x}, \mathbf{i}) &=
  ///   \begin{bmatrix}
  ///     f_x x'' + c_x
  /// \\  f_y y'' + c_y
  /// \\\end{bmatrix}
  /// \newline
  ///
  /// \\\begin{bmatrix}
  ///     x''
  /// \\  y''
  ///   \end{bmatrix} &=
  ///   \begin{bmatrix}
  ///     x' d + 2 p_1 x' y' + p_2 (r^2 + 2 x'^2)
  /// \\  y' d + 2 p_2 x' y' + p_1 (r^2 + 2 y'^2)
  /// \\\end{bmatrix}
  /// \newline
  ///
  /// \\d &= \frac{
  ///     1 + k_1 r^2 + k_2 r^4 + k_3 r^6
  ///   }{
  ///     1 + k_4 r^2 + k_5 r^4 + k_6 r^6
  ///   }
  /// \newline
  ///
  /// \\r &= x'^2 + y'^2
  /// \newline
  ///
  /// \\\begin{bmatrix}
  ///     x'
  /// \\  y'
  /// \\\end{bmatrix} &=
  ///   \begin{bmatrix}
  ///     x / z
  /// \\  y / z
  /// \\\end{bmatrix}
  /// \newline
  /// \f}
  ///
  /// A set of 3D points that results in valid projection is expressed as
  /// follows: \f{align}{
  ///    \Omega &= \{\mathbf{x} \in \mathbb{R}^3 ~|~ z > 0 \}
  /// \f}
  ///
  /// @param[in] p3d point to project
  /// @param[out] proj result of projection
  /// @param[out] d_proj_d_p3d if not nullptr computed Jacobian of projection
  /// with respect to p3d
  /// @param[out] d_proj_d_param point if not nullptr computed Jacobian of
  /// projection with respect to intrinsic parameters
  /// @return if projection is valid
  template <class DerivedPoint3D, class DerivedPoint2D,
            class DerivedJ3D = std::nullptr_t,
            class DerivedJparam = std::nullptr_t>
  inline bool project(const Eigen::MatrixBase<DerivedPoint3D>& p3d,
                      Eigen::MatrixBase<DerivedPoint2D>& proj,
                      DerivedJ3D d_proj_d_p3d = nullptr,
                      DerivedJparam d_proj_d_param = nullptr) const {
    checkProjectionDerivedTypes<DerivedPoint3D, DerivedPoint2D, DerivedJ3D,
                                DerivedJparam, N>();

    const typename EvalOrReference<DerivedPoint3D>::Type p3d_eval(p3d);

    const Scalar& fx = param_[0];
    const Scalar& fy = param_[1];
    const Scalar& cx = param_[2];
    const Scalar& cy = param_[3];
    const Scalar& k1 = param_[4];
    const Scalar& k2 = param_[5];
    const Scalar& p1 = param_[6];
    const Scalar& p2 = param_[7];
    const Scalar& k3 = param_[8];
    const Scalar& k4 = param_[9];
    const Scalar& k5 = param_[10];
    const Scalar& k6 = param_[11];

    const Scalar& x = p3d_eval[0];
    const Scalar& y = p3d_eval[1];
    const Scalar& z = p3d_eval[2];

    const Scalar xp = x / z;
    const Scalar yp = y / z;
    const Scalar r2 = xp * xp + yp * yp;
    const Scalar cdist = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) /
                         (1 + r2 * (k4 + r2 * (k5 + r2 * k6)));
    const Scalar deltaX = 2 * p1 * xp * yp + p2 * (r2 + 2 * xp * xp);
    const Scalar deltaY = 2 * p2 * xp * yp + p1 * (r2 + 2 * yp * yp);
    const Scalar xpp = xp * cdist + deltaX;
    const Scalar ypp = yp * cdist + deltaY;
    const Scalar u = fx * xpp + cx;
    const Scalar v = fy * ypp + cy;

    proj[0] = u;
    proj[1] = v;

    const bool is_valid = z >= Sophus::Constants<Scalar>::epsilonSqrt();

    // The following derivative formulas were computed automatically with sympy
    // (with `diff`, `simplify`, and `cse`) from the previous definition. Don't
    // try to understand them.

    if constexpr (!std::is_same_v<DerivedJ3D, std::nullptr_t>) {
      BASALT_ASSERT(d_proj_d_p3d);

      d_proj_d_p3d->setZero();

      // clang-format off
      const Scalar v0 = p1 * y;
      const Scalar v1 = p2 * x;
      const Scalar v3 = x * x;
      const Scalar v4 = y * y;
      const Scalar v5 = v3 + v4;
      const Scalar v7 = z * z;
      const Scalar v6 = v7 * v7;
      const Scalar v2 = v6 * v7;
      const Scalar v8 = k5 * v7;
      const Scalar v9 = k6 * v5;
      const Scalar v10 = k4 * v6 + v5 * (v8 + v9);
      const Scalar v11 = v10 * v5 + v2;
      const Scalar v12 = v11 * v11;
      const Scalar v13 = 2 * v12;
      const Scalar v14 = k2 * v7;
      const Scalar v15 = k3 * v5;
      const Scalar v16 = k1 * v6 + v5 * (v14 + v15);
      const Scalar v17 = v16 * v5 + v2;
      const Scalar v18 = v17 * z * (v10 + v5 * (v8 + 2 * v9));
      const Scalar v19 = 2 * v18;
      const Scalar v20 = v16 + v5 * (v14 + 2 * v15);
      const Scalar v21 = 2 * v20;
      const Scalar v22 = v11 * z;
      const Scalar v23 = 1 / v7;
      const Scalar v24 = 1 / v12;
      const Scalar v25 = fx * v24;
      const Scalar v26 = v23 * v25;
      const Scalar v27 = p2 * y;
      const Scalar v28 = x * y;
      const Scalar v29 = 2 * v12 * (p1 * x + v27) - 2 * v18 * v28 + 2 * v20 * v22 * v28;
      const Scalar v30 = 1 / (v7 * z);
      const Scalar v31 = 2 * x;
      const Scalar v32 = v22 * (v17 + v21 * v5);
      const Scalar v33 = fy * v24;
      const Scalar v34 = v23 * v33;

      const Scalar du_dx = v26 * (v13 * (v0 + 3 * v1) - v19 * v3 + v22 * (v17 + v21 * v3));
      const Scalar du_dy = v26 * v29;
      const Scalar du_dz = (-v25 * v30 * (v13 * (p2 * (3 * v3 + v4) + v0 * v31) - v18 * v31 * v5 + v32 * x));
      const Scalar dv_dx = v29 * v34;
      const Scalar dv_dy = v34 * (v13 * (3 * v0 + v1) - v19 * v4 + v22 * (v17 + v21 * v4));
      const Scalar dv_dz = (-v30 * v33 * (v13 * (p1 * (v3 + 3 * v4) + v27 * v31) - v19 * v5 * y + v32 * y));
      // clang-format on

      (*d_proj_d_p3d)(0, 0) = du_dx;
      (*d_proj_d_p3d)(0, 1) = du_dy;
      (*d_proj_d_p3d)(0, 2) = du_dz;
      (*d_proj_d_p3d)(1, 0) = dv_dx;
      (*d_proj_d_p3d)(1, 1) = dv_dy;
      (*d_proj_d_p3d)(1, 2) = dv_dz;
    } else {
      UNUSED(d_proj_d_p3d);
    }

    if constexpr (!std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(d_proj_d_param);
      d_proj_d_param->setZero();

      const Scalar w1 = x * x;
      const Scalar w2 = y * y;
      const Scalar w3 = w1 + w2;
      const Scalar w5 = z * z;
      const Scalar w4 = w5 * w5;
      const Scalar w0 = w4 * w5;
      const Scalar w6 = w0 + w3 * (k1 * w4 + w3 * (k2 * w5 + k3 * w3));
      const Scalar w7 = w6 * z;
      const Scalar w8 = w7 * x;
      const Scalar w9 = 2 * x * y;
      const Scalar w10 = 3 * w1 + w2;
      const Scalar w11 = w0 + w3 * (k4 * w4 + w3 * (k5 * w5 + k6 * w3));
      const Scalar w12 = 1 / w5;
      const Scalar w13 = 1 / w11;
      const Scalar w14 = w12 * w13;
      const Scalar w15 = w3 * z * w5;
      const Scalar w16 = fx * x;
      const Scalar w17 = w13 * w16;
      const Scalar w18 = w3 * w3;
      const Scalar w19 = w18 * z;
      const Scalar w20 = fx * w12;
      const Scalar w21 = w3 * w18 / z;
      const Scalar w22 = w13 * w13;
      const Scalar w23 = w22 * w6;
      const Scalar w24 = w16 * w23;
      const Scalar w25 = w18 * w22;
      const Scalar w26 = w7 * y;
      const Scalar w27 = w1 + 3 * w2;
      const Scalar w28 = fy * y;
      const Scalar w29 = w13 * w28;
      const Scalar w30 = fy * w12;
      const Scalar w31 = w23 * w28;
      const Scalar du_fx = w14 * (w11 * (p1 * w9 + p2 * w10) + w8);
      const Scalar du_fy = 0;
      const Scalar du_cx = 1;
      const Scalar du_cy = 0;
      const Scalar du_k1 = w15 * w17;
      const Scalar du_k2 = w17 * w19;
      const Scalar du_p1 = w20 * w9;
      const Scalar du_p2 = w10 * w20;
      const Scalar du_k3 = w17 * w21;
      const Scalar du_k4 = -w15 * w24;
      const Scalar du_k5 = -fx * w25 * w8;
      const Scalar du_k6 = -w21 * w24;
      const Scalar dv_fx = 0;
      const Scalar dv_fy = w14 * (w11 * (p1 * w27 + p2 * w9) + w26);
      const Scalar dv_cx = 0;
      const Scalar dv_cy = 1;
      const Scalar dv_k1 = w15 * w29;
      const Scalar dv_k2 = w19 * w29;
      const Scalar dv_p1 = w27 * w30;
      const Scalar dv_p2 = w30 * w9;
      const Scalar dv_k3 = w21 * w29;
      const Scalar dv_k4 = -w15 * w31;
      const Scalar dv_k5 = -fy * w25 * w26;
      const Scalar dv_k6 = -w21 * w31;

      (*d_proj_d_param)(0, 0) = du_fx;
      (*d_proj_d_param)(0, 1) = du_fy;
      (*d_proj_d_param)(0, 2) = du_cx;
      (*d_proj_d_param)(0, 3) = du_cy;
      (*d_proj_d_param)(0, 4) = du_k1;
      (*d_proj_d_param)(0, 5) = du_k2;
      (*d_proj_d_param)(0, 6) = du_p1;
      (*d_proj_d_param)(0, 7) = du_p2;
      (*d_proj_d_param)(0, 8) = du_k3;
      (*d_proj_d_param)(0, 9) = du_k4;
      (*d_proj_d_param)(0, 10) = du_k5;
      (*d_proj_d_param)(0, 11) = du_k6;
      (*d_proj_d_param)(1, 0) = dv_fx;
      (*d_proj_d_param)(1, 1) = dv_fy;
      (*d_proj_d_param)(1, 2) = dv_cx;
      (*d_proj_d_param)(1, 3) = dv_cy;
      (*d_proj_d_param)(1, 4) = dv_k1;
      (*d_proj_d_param)(1, 5) = dv_k2;
      (*d_proj_d_param)(1, 6) = dv_p1;
      (*d_proj_d_param)(1, 7) = dv_p2;
      (*d_proj_d_param)(1, 8) = dv_k3;
      (*d_proj_d_param)(1, 9) = dv_k4;
      (*d_proj_d_param)(1, 10) = dv_k5;
      (*d_proj_d_param)(1, 11) = dv_k6;
    } else {
      UNUSED(d_proj_d_param);
    }

    return is_valid;
  }

  /// @brief Distorts a normalized 2D point
  ///
  /// Given \f$ (x', y') \f$ computes \f$ (x'', y'') \f$ as defined @ref
  /// project. It can also optionally compute its jacobian.
  /// @param[in] undist Undistorted normalized 2D point \f$ (x', y') \f$
  /// @param[out] dist Result of distortion \f$ (x'', y'') \f$
  /// @param[out] d_dist_d_undist if not nullptr, computed Jacobian of @p dist
  /// w.r.t @p undist
  template <class DerivedJundist = std::nullptr_t>
  inline void distort(const Vec2& undist, Vec2& dist,
                      DerivedJundist d_dist_d_undist = nullptr) const {
    const Scalar& k1 = param_[4];
    const Scalar& k2 = param_[5];
    const Scalar& p1 = param_[6];
    const Scalar& p2 = param_[7];
    const Scalar& k3 = param_[8];
    const Scalar& k4 = param_[9];
    const Scalar& k5 = param_[10];
    const Scalar& k6 = param_[11];

    const Scalar xp = undist.x();
    const Scalar yp = undist.y();
    const Scalar r2 = xp * xp + yp * yp;
    const Scalar cdist = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) /
                         (1 + r2 * (k4 + r2 * (k5 + r2 * k6)));
    const Scalar deltaX = 2 * p1 * xp * yp + p2 * (r2 + 2 * xp * xp);
    const Scalar deltaY = 2 * p2 * xp * yp + p1 * (r2 + 2 * yp * yp);
    const Scalar xpp = xp * cdist + deltaX;
    const Scalar ypp = yp * cdist + deltaY;
    dist.x() = xpp;
    dist.y() = ypp;

    if constexpr (!std::is_same_v<DerivedJundist, std::nullptr_t>) {
      BASALT_ASSERT(d_dist_d_undist);

      // Expressions derived with sympy
      const Scalar v0 = xp * xp;
      const Scalar v1 = yp * yp;
      const Scalar v2 = v0 + v1;
      const Scalar v3 = k6 * v2;
      const Scalar v4 = k4 + v2 * (k5 + v3);
      const Scalar v5 = v2 * v4 + 1;
      const Scalar v6 = v5 * v5;
      const Scalar v7 = 1 / v6;
      const Scalar v8 = p1 * yp;
      const Scalar v9 = p2 * xp;
      const Scalar v10 = 2 * v6;
      const Scalar v11 = k3 * v2;
      const Scalar v12 = k1 + v2 * (k2 + v11);
      const Scalar v13 = v12 * v2 + 1;
      const Scalar v14 = v13 * (v2 * (k5 + 2 * v3) + v4);
      const Scalar v15 = 2 * v14;
      const Scalar v16 = v12 + v2 * (k2 + 2 * v11);
      const Scalar v17 = 2 * v16;
      const Scalar v18 = xp * yp;
      const Scalar v19 =
          2 * v7 * (-v14 * v18 + v16 * v18 * v5 + v6 * (p1 * xp + p2 * yp));

      const Scalar dxpp_dxp =
          v7 * (-v0 * v15 + v10 * (v8 + 3 * v9) + v5 * (v0 * v17 + v13));
      const Scalar dxpp_dyp = v19;
      const Scalar dypp_dxp = v19;
      const Scalar dypp_dyp =
          v7 * (-v1 * v15 + v10 * (3 * v8 + v9) + v5 * (v1 * v17 + v13));

      (*d_dist_d_undist)(0, 0) = dxpp_dxp;
      (*d_dist_d_undist)(0, 1) = dxpp_dyp;
      (*d_dist_d_undist)(1, 0) = dypp_dxp;
      (*d_dist_d_undist)(1, 1) = dypp_dyp;
    } else {
      UNUSED(d_dist_d_undist);
    }
  }

  /// @brief Unproject the point
  /// @note Computing the jacobians is not implemented
  ///
  /// The unprojection method is based on OpenCV implementation of
  /// undistortPoints. It uses a Jacobi-like solver with \f$N = 5\f$ iterations.
  /// Tests for project-unproject inversion for this function are disabled
  /// because sometimes the unprojected result is indeed a fixed point, but not
  /// the original one passed as argument to @ref project.
  ///
  /// The unprojection function is computed as follows:
  /// \f{align}{
  ///   \pi^{-1}(\mathbf{u}, \mathbf{i}) &= \hat{\mathbf{x}}^*
  ///   \newline
  ///
  /// \\\hat{\mathbf{x}}^* &=
  ///   \frac{1}{\mathbf{x}_x^{*2} + \mathbf{x}_y^{*2} + 1}
  ///   \begin{bmatrix} \mathbf{x}_x^* \\ \mathbf{x}_y^* \\ 1 \\ \end{bmatrix}
  ///   \newline
  ///
  /// \\\mathbf{x}_* &= \mathbf{x}_N
  ///   \newline
  ///
  /// \\\mathbf{x}_0 &=
  ///   \begin{bmatrix}
  ///     (u - c_x) / f_x
  /// \\  (v - c_y) / f_y
  /// \\\end{bmatrix}
  ///   \newline
  ///
  /// \\\mathbf{x}_{n+1} &=
  ///   \frac{\mathbf{x}_0 - \mathbf{\Delta}(\mathbf{x}_n)}{d(\mathbf{x}_n)}
  ///   \newline
  ///
  /// \\\mathbf{\Delta}(x, y) &= \begin{bmatrix}
  ///     2 p_1 x y + p_2 (r^2 + 2 x^2)
  /// \\  2 p_2 x y + p_1 (r^2 + 2 y^2)
  /// \\\end{bmatrix}
  ///   \newline
  ///
  /// \\d(x, y) &= \frac{
  ///     1 + k_1 r^2 + k_2 r^4 + k_3 r^6
  ///   }{
  ///     1 + k_4 r^2 + k_5 r^4 + k_6 r^6
  ///   }
  ///   \newline
  ///
  /// \\r &= \sqrt{x^2 + y^2}
  ///   \newline
  ///
  /// \\N &= 5
  /// \f}
  ///
  /// @param[in] proj point to unproject
  /// @param[out] p3d result of unprojection
  /// @param[out] d_p3d_d_proj \b UNIMPLEMENTED if not nullptr, computed
  /// Jacobian of unprojection with respect to proj
  /// @param[out] d_p3d_d_param \b UNIMPLEMENTED if not nullptr, computed
  /// Jacobian of unprojection with respect to intrinsic parameters
  /// @return if unprojection is valid
  template <class DerivedPoint2D, class DerivedPoint3D,
            class DerivedJ2D = std::nullptr_t,
            class DerivedJparam = std::nullptr_t>
  inline bool unproject(const Eigen::MatrixBase<DerivedPoint2D>& proj,
                        Eigen::MatrixBase<DerivedPoint3D>& p3d,
                        DerivedJ2D d_p3d_d_proj = nullptr,
                        DerivedJparam d_p3d_d_param = nullptr) const {
    checkUnprojectionDerivedTypes<DerivedPoint2D, DerivedPoint3D, DerivedJ2D,
                                  DerivedJparam, N>();
    const typename EvalOrReference<DerivedPoint2D>::Type proj_eval(proj);

    const Scalar& fx = param_[0];
    const Scalar& fy = param_[1];
    const Scalar& cx = param_[2];
    const Scalar& cy = param_[3];

    const Scalar& u = proj_eval[0];
    const Scalar& v = proj_eval[1];

    const Scalar x0 = (u - cx) / fx;
    const Scalar y0 = (v - cy) / fy;

    // Newton solver
    Vec2 dist{x0, y0};
    Vec2 undist{dist};
    const Scalar EPS = Sophus::Constants<Scalar>::epsilonSqrt();
    constexpr int N = 20;  // Max iterations
    for (int i = 0; i < N; i++) {
      Mat22 J{};
      Vec2 fundist{};
      distort(undist, fundist, &J);
      Vec2 residual = fundist - dist;
      undist -= J.inverse() * residual;
      if (residual.squaredNorm() < EPS) {
        break;
      }
    }

    const Scalar mx = undist.x();
    const Scalar my = undist.y();
    const Scalar norm_inv = 1 / sqrt(mx * mx + my * my + 1);
    p3d.setZero();
    p3d[0] = mx * norm_inv;
    p3d[1] = my * norm_inv;
    p3d[2] = norm_inv;

    if constexpr (!std::is_same_v<DerivedJ2D, std::nullptr_t> ||
                  !std::is_same_v<DerivedJparam, std::nullptr_t>) {
      BASALT_ASSERT(false);  // Not implemented
      // If this gets implemented update: docs, benchmarks and tests
    }
    UNUSED(d_p3d_d_proj);
    UNUSED(d_p3d_d_param);

    return true;
  }

  /// @brief Set parameters from initialization
  ///
  /// Initializes the camera model to  \f$ \left[
  /// f_x, f_y, c_x, c_y, 0, 0, 0, 0, 0, 0, 0, 0 \right]^T \f$
  /// @param[in] init vector [f_x, f_y, c_x, c_y]
  inline void setFromInit(const Vec4& init) {
    param_.setZero();
    param_[0] = init[0];
    param_[1] = init[1];
    param_[2] = init[2];
    param_[3] = init[3];
  }

  /// @brief Increment intrinsic parameters by inc
  ///
  /// @param[in] inc increment vector
  void operator+=(const VecN& inc) { param_ += inc; }

  /// @brief Returns a const reference to the intrinsic parameters vector
  ///
  /// The order is following: \f$ \left[
  /// f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6
  /// \right]^T \f$
  /// @return const reference to the intrinsic parameters vector
  const VecN& getParam() const { return param_; }

  /// @brief Projections used for unit-tests
  static Eigen::aligned_vector<PinholeRadtan8Camera> getTestProjections() {
    Eigen::aligned_vector<PinholeRadtan8Camera> res;

    VecN vec1{};

    vec1 << 269.0600776672363, 269.1679859161377, 324.3333053588867,
        245.22674560546875, 0.6257319450378418, 0.46612036228179932,
        -0.00018502399325370789, -4.2882973502855748e-5, 0.0041795829311013222,
        0.89431935548782349, 0.54253977537155151, 0.0662121474742889;
    res.emplace_back(vec1);

    return res;
  }

  /// @brief Resolutions used for unit-tests
  static Eigen::aligned_vector<Eigen::Vector2i> getTestResolutions() {
    Eigen::aligned_vector<Eigen::Vector2i> res;
    res.emplace_back(640, 480);
    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param_;
};

}  // namespace basalt

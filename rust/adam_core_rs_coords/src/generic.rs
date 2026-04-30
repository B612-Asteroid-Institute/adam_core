//! Generic coordinate-representation kernels over the `Scalar` trait.
//!
//! Each kernel returns the 6 canonical elements of the target representation
//! (the elements stored in the corresponding `quivr.Table`). They are the
//! building blocks used both by the batch f64 fast paths in this crate and by
//! Jacobian evaluation (via `Dual<6>` inputs) for covariance propagation.

use adam_core_rs_autodiff::Scalar;

pub const TWO_PI_F64: f64 = std::f64::consts::PI * 2.0;
pub const RAD2DEG_F64: f64 = 180.0 / std::f64::consts::PI;
const OBLIQUITY_RAD_F64: f64 = 84381.448_f64 * std::f64::consts::PI / (180.0_f64 * 3600.0_f64);

#[inline]
fn deg<T: Scalar>() -> T {
    T::from_f64(RAD2DEG_F64)
}

#[inline]
fn two_pi<T: Scalar>() -> T {
    T::from_f64(TWO_PI_F64)
}

#[inline]
fn normalize_lon<T: Scalar>(lon: T) -> T {
    if lon.re() < 0.0 {
        lon + two_pi::<T>()
    } else {
        lon
    }
}

pub fn cartesian_to_spherical6<T: Scalar>(v: &[T; 6]) -> [T; 6] {
    let x = v[0];
    let y = v[1];
    let z = v[2];
    let vx = v[3];
    let vy = v[4];
    let vz = v[5];

    let zero = T::from_f64(0.0);
    let rho = (x * x + y * y + z * z).sqrt();
    let lon = normalize_lon(y.atan2(x));
    let lat = if rho.re() == 0.0 {
        zero
    } else {
        (z / rho).asin()
    };

    let vrho = if rho.re() == 0.0 {
        zero
    } else {
        (x * vx + y * vy + z * vz) / rho
    };
    let r2xy = x * x + y * y;
    let vlon = if x.re() == 0.0 && y.re() == 0.0 {
        zero
    } else {
        (vy * x - vx * y) / r2xy
    };
    let vlat = if (x.re() == 0.0 && y.re() == 0.0) || rho.re() == 0.0 {
        zero
    } else {
        (vz - vrho * z / rho) / r2xy.sqrt()
    };

    let k = deg::<T>();
    [rho, lon * k, lat * k, vrho, vlon * k, vlat * k]
}

pub fn spherical_to_cartesian6<T: Scalar>(v: &[T; 6]) -> [T; 6] {
    let inv_deg = T::from_f64(1.0 / RAD2DEG_F64);
    let rho = v[0];
    let lon = v[1] * inv_deg;
    let lat = v[2] * inv_deg;
    let vrho = v[3];
    let vlon = v[4] * inv_deg;
    let vlat = v[5] * inv_deg;

    let (s_lat, c_lat) = lat.sin_cos();
    let (s_lon, c_lon) = lon.sin_cos();

    let x = rho * c_lat * c_lon;
    let y = rho * c_lat * s_lon;
    let z = rho * s_lat;

    let vx = c_lat * c_lon * vrho - rho * c_lat * s_lon * vlon - rho * s_lat * c_lon * vlat;
    let vy = c_lat * s_lon * vrho + rho * c_lat * c_lon * vlon - rho * s_lat * s_lon * vlat;
    let vz = s_lat * vrho + rho * c_lat * vlat;

    [x, y, z, vx, vy, vz]
}

pub fn cartesian_to_geodetic6<T: Scalar>(v: &[T; 6], a: f64, f: f64) -> [T; 6] {
    let x = v[0];
    let y = v[1];
    let z = v[2];
    let vx = v[3];
    let vy = v[4];
    let vz = v[5];

    let a_t = T::from_f64(a);
    let b_t = T::from_f64(a * (1.0 - f));
    let e2 = (a_t * a_t - b_t * b_t) / (a_t * a_t);
    let one = T::from_f64(1.0);
    let xy = (x * x + y * y).sqrt();

    let lat0 = z.atan2(xy * (one - e2));
    let sin_lat0 = lat0.sin();
    let cos_lat0 = lat0.cos();
    let n0 = a_t / (one - e2 * sin_lat0 * sin_lat0).sqrt();
    let alt0 = xy / cos_lat0 - n0;

    let lat = z.atan2(xy * (one - e2 * n0 / (n0 + alt0)));
    let lon = normalize_lon(y.atan2(x));
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    let n = a_t / (one - e2 * sin_lat * sin_lat).sqrt();
    let alt = xy / cos_lat - n;

    let v_east = -sin_lon * vx + cos_lon * vy;
    let v_north = -sin_lat * cos_lon * vx - sin_lat * sin_lon * vy + cos_lat * vz;
    let v_up = cos_lat * cos_lon * vx + cos_lat * sin_lon * vy + sin_lat * vz;

    let k = deg::<T>();
    [alt, lon * k, lat * k, v_east * k, v_north * k, v_up]
}

fn cross3<T: Scalar>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3<T: Scalar>(a: [T; 3], b: [T; 3]) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3<T: Scalar>(v: [T; 3]) -> T {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn clamp_unit<T: Scalar>(x: T) -> T {
    x.max(T::from_f64(-1.0)).min(T::from_f64(1.0))
}

fn calc_mean_anomaly<T: Scalar>(nu: T, e: T) -> T {
    let two_pi = two_pi::<T>();
    let nu_norm = if nu.re() >= TWO_PI_F64 {
        nu - two_pi
    } else {
        nu
    };
    let one = T::from_f64(1.0);
    let two = T::from_f64(2.0);
    if e.re() < 1.0 {
        let ecc = ((one - e * e).sqrt() * nu_norm.sin()).atan2(e + nu_norm.cos());
        let m = ecc - e * ecc.sin();
        if m.re() < 0.0 {
            m + two_pi
        } else {
            m
        }
    } else if e.re() > 1.0 {
        let hyp = two * (((e - one) / (e + one)).sqrt() * (nu_norm / two).tan()).atanh();
        let m = e * hyp.sinh() - hyp;
        if m.re() < 0.0 {
            m + two_pi
        } else {
            m
        }
    } else {
        let d = (nu_norm / two).atan();
        let m = d + d * d * d / T::from_f64(3.0);
        if m.re() < 0.0 {
            m + two_pi
        } else {
            m
        }
    }
}

fn solve_barker<T: Scalar>(m: T) -> T {
    let three = T::from_f64(3.0);
    let nine = T::from_f64(9.0);
    let one = T::from_f64(1.0);
    let two = T::from_f64(2.0);
    let term = three * m + (nine * m * m + one).sqrt();
    // term^(1/3) - term^(-1/3)
    let t_cbrt = term.cbrt();
    let inv = one / t_cbrt;
    two * (t_cbrt - inv).atan()
}

fn solve_kepler_true_anomaly<T: Scalar>(e: T, m: T, max_iter: usize, tol: f64) -> T {
    let one = T::from_f64(1.0);
    let two = T::from_f64(2.0);

    if e.re() < 1.0 {
        let mut ecc = m;
        let mut iter = 0_usize;
        loop {
            let (sin_e, cos_e) = ecc.sin_cos();
            let f = ecc - e * sin_e - m;
            let fp = one - e * cos_e;
            let step = f / fp;
            ecc -= step;
            iter += 1;
            if step.re().abs() <= tol || iter > max_iter {
                break;
            }
        }
        let nu = two
            * ((one + e).sqrt() * (ecc / two).sin()).atan2((one - e).sqrt() * (ecc / two).cos());
        wrap_two_pi(nu)
    } else if e.re() > 1.0 {
        let mut hyp = m;
        let mut iter = 0_usize;
        loop {
            let f = e * hyp.sinh() - hyp - m;
            let fp = e * hyp.cosh() - one;
            let step = f / fp;
            hyp -= step;
            iter += 1;
            if step.re().abs() <= tol || iter > max_iter {
                break;
            }
        }
        let nu = two
            * (((e + one).sqrt() * (hyp / two).sinh()) / ((e - one).sqrt() * (hyp / two).cosh()))
                .atan();
        wrap_two_pi(nu)
    } else {
        wrap_two_pi(solve_barker(m))
    }
}

fn wrap_two_pi<T: Scalar>(nu: T) -> T {
    let two_pi = two_pi::<T>();
    let mut out = nu;
    if out.re() < 0.0 {
        out += two_pi;
    }
    if out.re() >= TWO_PI_F64 {
        out -= two_pi;
    }
    out
}

/// Converts (a, e, i[deg], raan[deg], ap[deg], M[deg]) and mu to (x, y, z, vx, vy, vz).
pub fn keplerian_to_cartesian6<T: Scalar>(v: &[T; 6], mu: T, max_iter: usize, tol: f64) -> [T; 6] {
    let inv_deg = T::from_f64(1.0 / RAD2DEG_F64);
    let a = v[0];
    let e = v[1];
    let i = v[2] * inv_deg;
    let raan = v[3] * inv_deg;
    let ap = v[4] * inv_deg;
    let m = v[5] * inv_deg;

    let one = T::from_f64(1.0);
    let p = a * (one - e * e);
    let nu = solve_kepler_true_anomaly(e, m, max_iter, tol);

    let (sin_nu, cos_nu) = nu.sin_cos();
    let denom = one + e * cos_nu;
    let r_pqw = [p * cos_nu / denom, p * sin_nu / denom, T::from_f64(0.0)];
    let sqrt_mu_over_p = (mu / p).sqrt();
    let v_pqw = [
        -sqrt_mu_over_p * sin_nu,
        sqrt_mu_over_p * (e + cos_nu),
        T::from_f64(0.0),
    ];

    let (sin_raan, cos_raan) = raan.sin_cos();
    let (sin_ap, cos_ap) = ap.sin_cos();
    let (sin_i, cos_i) = i.sin_cos();

    // rotation = R_z(-raan) * R_x(-i) * R_z(-ap) applied to PQW
    // Combined 3x3 PQW->IJK matrix:
    let r11 = cos_raan * cos_ap - sin_raan * cos_i * sin_ap;
    let r12 = -cos_raan * sin_ap - sin_raan * cos_i * cos_ap;
    let r21 = sin_raan * cos_ap + cos_raan * cos_i * sin_ap;
    let r22 = -sin_raan * sin_ap + cos_raan * cos_i * cos_ap;
    let r31 = sin_i * sin_ap;
    let r32 = sin_i * cos_ap;

    let rx = r11 * r_pqw[0] + r12 * r_pqw[1];
    let ry = r21 * r_pqw[0] + r22 * r_pqw[1];
    let rz = r31 * r_pqw[0] + r32 * r_pqw[1];
    let vx = r11 * v_pqw[0] + r12 * v_pqw[1];
    let vy = r21 * v_pqw[0] + r22 * v_pqw[1];
    let vz = r31 * v_pqw[0] + r32 * v_pqw[1];

    [rx, ry, rz, vx, vy, vz]
}

/// Converts Cartesian (x, y, z, vx, vy, vz) and mu to 6 Keplerian elements
/// (a, e, i[deg], raan[deg], ap[deg], M[deg]). Elliptic and hyperbolic cases
/// only; parabolic inputs return NaN for `a` matching legacy behavior.
pub fn cartesian_to_keplerian6<T: Scalar>(v: &[T; 6], mu: T) -> [T; 6] {
    let float_tol = 1e-15_f64;
    let zero = T::from_f64(0.0);
    let one = T::from_f64(1.0);
    let two = T::from_f64(2.0);
    let two_pi = two_pi::<T>();

    let r = [v[0], v[1], v[2]];
    let vel = [v[3], v[4], v[5]];

    let r_mag = norm3(r);
    let v_mag = norm3(vel);
    let sme = v_mag * v_mag / two - mu / r_mag;

    let h = cross3(r, vel);
    let h_mag = norm3(h);
    let n_vec = cross3([zero, zero, one], h);
    let n_mag = norm3(n_vec);

    let rv = dot3(r, vel);
    let scale = v_mag * v_mag - mu / r_mag;
    let e_vec = [
        (scale * r[0] - rv * vel[0]) / mu,
        (scale * r[1] - rv * vel[1]) / mu,
        (scale * r[2] - rv * vel[2]) / mu,
    ];
    let e = norm3(e_vec);

    let i = clamp_unit(h[2] / h_mag).acos();

    let mut raan = if n_mag.re() == 0.0 {
        zero
    } else {
        clamp_unit(n_vec[0] / n_mag).acos()
    };
    if n_vec[1].re() < 0.0 {
        raan = two_pi - raan;
    }
    if i.re() < float_tol || (i.re() - TWO_PI_F64).abs() < float_tol {
        raan = zero;
    }

    let mut ap = if n_mag.re() == 0.0 || e.re() == 0.0 {
        zero
    } else {
        clamp_unit(dot3(n_vec, e_vec) / (n_mag * e)).acos()
    };
    if e_vec[2].re() < 0.0 {
        ap = two_pi - ap;
    }
    if e.re().abs() < float_tol {
        ap = zero;
    }

    let mut nu = if e.re() == 0.0 {
        zero
    } else {
        clamp_unit(dot3(e_vec, r) / (e * r_mag)).acos()
    };
    if rv.re() < 0.0 {
        nu = two_pi - nu;
    }

    let a = mu / (-two * sme);
    let m_anom = calc_mean_anomaly(nu, e);
    let k = deg::<T>();
    [a, e, i * k, raan * k, ap * k, m_anom * k]
}

/// Converts Cartesian to Cometary (q, e, i[deg], raan[deg], ap[deg], tp) matching the
/// legacy JAX path: tp is placed so that |tp - t0| <= period/2 for elliptic orbits.
pub fn cartesian_to_cometary6<T: Scalar>(v: &[T; 6], t0: T, mu: T) -> [T; 6] {
    let kep = cartesian_to_keplerian6(v, mu);
    let a = kep[0];
    let e = kep[1];
    let i_deg = kep[2];
    let raan_deg = kep[3];
    let ap_deg = kep[4];
    let m_deg = kep[5];

    let one = T::from_f64(1.0);
    let q = a * (one - e);
    let inv_deg = T::from_f64(1.0 / RAD2DEG_F64);
    let m_rad = m_deg * inv_deg;
    let n = (mu / a.abs().powi(3)).sqrt();
    let period = two_pi::<T>() / n;
    let dtp = if m_rad.re() > std::f64::consts::PI && e.re() < 1.0 {
        period - m_rad / n
    } else {
        -(m_rad / n)
    };
    let tp = t0 + dtp;
    [q, e, i_deg, raan_deg, ap_deg, tp]
}

/// Converts Cometary (q, e, i[deg], raan[deg], ap[deg], tp) to Cartesian matching
/// legacy JAX path: the mean anomaly is wrapped via the same branch as the legacy
/// `jnp.where(dtp > 0, 2π - dtp*n, -dtp*n)` rule.
pub fn cometary_to_cartesian6<T: Scalar>(
    v: &[T; 6],
    t0: T,
    mu: T,
    max_iter: usize,
    tol: f64,
) -> [T; 6] {
    let q = v[0];
    let e = v[1];
    let i_deg = v[2];
    let raan_deg = v[3];
    let ap_deg = v[4];
    let tp = v[5];

    let one = T::from_f64(1.0);
    let a = q / (one - e);
    let n = (mu / a.abs().powi(3)).sqrt();
    let dtp = tp - t0;
    let two_pi = two_pi::<T>();
    let m_rad = if dtp.re() > 0.0 {
        two_pi - dtp * n
    } else {
        -(dtp * n)
    };
    let m_deg = m_rad * deg::<T>();
    let kep = [a, e, i_deg, raan_deg, ap_deg, m_deg];
    keplerian_to_cartesian6(&kep, mu, max_iter, tol)
}

pub fn rotate_equatorial_to_ecliptic6<T: Scalar>(v: &[T; 6]) -> [T; 6] {
    let cos_o = T::from_f64(OBLIQUITY_RAD_F64.cos());
    let sin_o = T::from_f64(OBLIQUITY_RAD_F64.sin());
    [
        v[0],
        cos_o * v[1] + sin_o * v[2],
        -sin_o * v[1] + cos_o * v[2],
        v[3],
        cos_o * v[4] + sin_o * v[5],
        -sin_o * v[4] + cos_o * v[5],
    ]
}

pub fn rotate_ecliptic_to_equatorial6<T: Scalar>(v: &[T; 6]) -> [T; 6] {
    let cos_o = T::from_f64(OBLIQUITY_RAD_F64.cos());
    let sin_o = T::from_f64(OBLIQUITY_RAD_F64.sin());
    [
        v[0],
        cos_o * v[1] - sin_o * v[2],
        sin_o * v[1] + cos_o * v[2],
        v[3],
        cos_o * v[4] - sin_o * v[5],
        sin_o * v[4] + cos_o * v[5],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use adam_core_rs_autodiff::Dual;

    const ROW_F64: [f64; 6] = [1.5, 0.1, 0.2, 0.003, 0.004, 0.005];

    fn approx(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() <= tol * (1.0 + b.abs()),
            "expected {a} ~= {b} (tol {tol})"
        );
    }

    #[test]
    fn cartesian_to_spherical_f64_and_dual_values_agree() {
        let rf = cartesian_to_spherical6::<f64>(&ROW_F64);
        let rows_d: [Dual<6>; 6] = Dual::seed(ROW_F64);
        let rd = cartesian_to_spherical6::<Dual<6>>(&rows_d);
        for i in 0..6 {
            approx(rd[i].re, rf[i], 1e-15);
        }
    }

    #[test]
    fn jacobian_roundtrip_cartesian_spherical_is_identity() {
        // J(C<-S(C)) should be identity (at any point where the transform is invertible).
        let rows_d: [Dual<6>; 6] = Dual::seed(ROW_F64);
        let spherical = cartesian_to_spherical6::<Dual<6>>(&rows_d);
        let back = spherical_to_cartesian6::<Dual<6>>(&spherical);
        for i in 0..6 {
            approx(back[i].re, ROW_F64[i], 1e-10);
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                approx(back[i].du[j], expected, 1e-8);
            }
        }
    }

    #[test]
    fn keplerian_roundtrip_preserves_input() {
        let kep: [f64; 6] = [2.5, 0.15, 20.0, 45.0, 30.0, 60.0];
        let mu = 2.95912208284120e-04_f64;
        let rows_d: [Dual<6>; 6] = Dual::seed(kep);
        let mu_d = Dual::constant(mu);
        let cart = keplerian_to_cartesian6::<Dual<6>>(&rows_d, mu_d, 100, 1e-15);
        let back = cartesian_to_keplerian6::<Dual<6>>(&cart, mu_d);
        for i in 0..6 {
            approx(back[i].re, kep[i], 1e-10);
        }
        // Jacobian of roundtrip should be identity (well-conditioned orbit).
        for (i, bi) in back.iter().enumerate() {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                approx(bi.du[j], expected, 1e-6);
            }
        }
    }

    #[test]
    fn cartesian_to_spherical_matches_existing_f64() {
        use crate::cartesian_to_spherical_row;
        let rows: &[[f64; 6]] = &[
            [1.0, 0.5, 0.3, 0.01, 0.02, -0.005],
            [-2.3, 1.8, 0.9, 0.004, -0.007, 0.003],
            [0.8, -0.4, -0.2, -0.001, 0.006, -0.004],
        ];
        for row in rows {
            let a = cartesian_to_spherical6::<f64>(row);
            let b = cartesian_to_spherical_row(row);
            for i in 0..6 {
                approx(a[i], b[i], 1e-14);
            }
        }
    }

    #[test]
    fn keplerian_to_cartesian_matches_existing_f64() {
        use crate::keplerian_to_cartesian_a_row;
        let mu = 2.95912208284120e-04_f64;
        let rows: &[[f64; 6]] = &[
            [2.5, 0.15, 20.0, 45.0, 30.0, 60.0],
            [1.2, 0.3, 5.0, 100.0, 250.0, 10.0],
            [5.2, 0.05, 1.3, 15.0, 270.0, 180.0],
        ];
        for row in rows {
            let a = keplerian_to_cartesian6::<f64>(row, mu, 100, 1e-15);
            let b = keplerian_to_cartesian_a_row(row, mu, 100, 1e-15);
            for i in 0..6 {
                approx(a[i], b[i], 1e-10);
            }
        }
    }

    #[test]
    fn cometary_roundtrip_preserves_input_and_jacobian() {
        let com: [f64; 6] = [0.9, 0.6, 10.0, 80.0, 40.0, 59000.0];
        let t0 = 59100.0_f64;
        let mu = 2.95912208284120e-04_f64;
        let rows_d: [Dual<6>; 6] = Dual::seed(com);
        let t0_d = Dual::constant(t0);
        let mu_d = Dual::constant(mu);
        let cart = cometary_to_cartesian6::<Dual<6>>(&rows_d, t0_d, mu_d, 100, 1e-15);
        let back = cartesian_to_cometary6::<Dual<6>>(&cart, t0_d, mu_d);
        for i in 0..6 {
            approx(back[i].re, com[i], 1e-8);
        }
        for (i, bi) in back.iter().enumerate() {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                approx(bi.du[j], expected, 1e-4);
            }
        }
    }

    #[test]
    fn rotate_roundtrip_eq2ec_jacobian_is_identity() {
        let rows_d: [Dual<6>; 6] = Dual::seed(ROW_F64);
        let ec = rotate_equatorial_to_ecliptic6::<Dual<6>>(&rows_d);
        let back = rotate_ecliptic_to_equatorial6::<Dual<6>>(&ec);
        for i in 0..6 {
            approx(back[i].re, ROW_F64[i], 1e-15);
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                approx(back[i].du[j], expected, 1e-14);
            }
        }
    }
}
